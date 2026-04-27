import * as vscode from "vscode";
import { randomUUID } from "node:crypto";
import { appendFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { gzipSync, gunzipSync } from "node:zlib";
import {
	CancellationToken,
	LanguageModelChatInformation,
	LanguageModelChatProvider,
	LanguageModelChatRequestMessage,
	ProvideLanguageModelChatResponseOptions,
	LanguageModelResponsePart2,
	Progress,
} from "vscode";

import type { HFModelItem } from "./types";

import type { OllamaRequestBody } from "./ollama/ollamaTypes";

import {
	parseModelId,
	createRetryConfig,
	executeWithRetry,
	normalizeUserModels,
	mapRole,
	isToolResultPart,
	collectToolResultText,
	isInternalMarkerMimeType,
} from "./utils";

import { prepareLanguageModelChatInformation } from "./provideModel";
import { countMessageTokens } from "./provideToken";
import { updateContextStatusBar } from "./statusBar";
import { OllamaApi } from "./ollama/ollamaApi";
import { OpenaiApi } from "./openai/openaiApi";
import { OpenaiResponsesApi } from "./openai/openaiResponsesApi";
import { AnthropicApi } from "./anthropic/anthropicApi";
import { AnthropicRequestBody } from "./anthropic/anthropicTypes";
import { GeminiApi, buildGeminiGenerateContentUrl, type GeminiToolCallMeta } from "./gemini/geminiApi";
import type { GeminiGenerateContentRequest } from "./gemini/geminiTypes";
import { CommonApi } from "./commonApi";
import { logger } from "./logger";

const DEBUG_LOG_PATH = path.join(tmpdir(), "oaicopilot-deepseek-debug.log");
const DEBUG_LOG_ENABLED = process.env.OAICOPILOT_DEBUG_LOG === "1";

function safeDebugStringify(value: unknown): string {
	try {
		return JSON.stringify(value, null, 2);
	} catch (error) {
		return JSON.stringify({
			stringifyError: error instanceof Error ? error.message : String(error),
		});
	}
}

function summarizeChatRequestMessages(messages: readonly LanguageModelChatRequestMessage[]): Array<Record<string, unknown>> {
	return messages.map((message, index) => {
		const summary: Record<string, unknown> = {
			index,
			role: mapRole(message),
			partTypes: [],
			text: "",
			thinking: "",
			toolCalls: [],
			toolResults: [],
			markers: [],
		};
		for (const part of message.content ?? []) {
			if (part instanceof vscode.LanguageModelTextPart) {
				summary.partTypes = [...(summary.partTypes as unknown[]), "text"];
				summary.text = `${summary.text ?? ""}${part.value}`;
			} else if (part instanceof vscode.LanguageModelThinkingPart) {
				const value = Array.isArray(part.value) ? part.value.join("") : part.value;
				summary.partTypes = [...(summary.partTypes as unknown[]), "thinking"];
				summary.thinking = `${summary.thinking ?? ""}${value}`;
			} else if (part instanceof vscode.LanguageModelToolCallPart) {
				summary.partTypes = [...(summary.partTypes as unknown[]), "tool_call"];
				(summary.toolCalls as unknown[]).push({
					callId: part.callId,
					name: part.name,
					input: part.input ?? {},
				});
			} else if (isToolResultPart(part)) {
				summary.partTypes = [...(summary.partTypes as unknown[]), "tool_result"];
				(summary.toolResults as unknown[]).push({
					callId: part.callId,
					content: collectToolResultText(part),
				});
			} else if (part instanceof vscode.LanguageModelDataPart) {
				summary.partTypes = [...(summary.partTypes as unknown[]), `data:${part.mimeType}`];
				if (isInternalMarkerMimeType(part.mimeType)) {
					(summary.markers as unknown[]).push(part.mimeType);
				}
			}
		}
		return summary;
	});
}

function summarizeOpenAIChatMessages(messages: readonly unknown[]): Array<Record<string, unknown>> {
	return messages.map((message, index) => {
		const normalized = (message ?? {}) as {
			role?: unknown;
			content?: unknown;
			reasoning_content?: unknown;
			tool_calls?: unknown;
			tool_call_id?: unknown;
		};
		return {
		index,
		role: normalized.role,
		content: normalized.content,
		reasoning_content: normalized.reasoning_content,
		tool_calls: normalized.tool_calls,
		tool_call_id: normalized.tool_call_id,
		};
	});
}

async function appendDeepSeekDebugLog(event: string, payload: Record<string, unknown>): Promise<void> {
	if (!DEBUG_LOG_ENABLED) {
		return;
	}
	const line = [
		`=== ${new Date().toISOString()} ${event} ===`,
		safeDebugStringify(payload),
		"",
	].join("\n");
	try {
		await appendFile(DEBUG_LOG_PATH, line, "utf8");
	} catch (error) {
		console.error("[OAI Compatible Model Provider] Failed to write debug log", {
			path: DEBUG_LOG_PATH,
			error: error instanceof Error ? error.message : String(error),
		});
	}
}

/**
 * VS Code Chat provider backed by Hugging Face Inference Providers.
 */
export class HuggingFaceChatModelProvider implements LanguageModelChatProvider {
	/** Track last request completion time for delay calculation. */
	private _lastRequestTime: number | null = null;

	private readonly _geminiToolCallMetaByCallId = new Map<string, GeminiToolCallMeta>();
	private readonly _openaiResponsesPreviousResponseIdUnsupportedBaseUrls = new Set<string>();
	private readonly _openaiChatReasoningCache: PersistentOpenAIChatReasoningCache;

	static readonly OPENAI_RESPONSES_STATEFUL_MARKER_MIME = "application/vnd.oaicopilot.stateful-marker";
	static readonly OPENAI_CHAT_STATE_MARKER_MIME = "application/vnd.oaicopilot.openai-chat-state";
	static readonly OPENAI_CHAT_REASONING_MARKER_MIME = "application/vnd.oaicopilot.openai-chat-reasoning";

	/**
	 * Create a provider using the given secret storage for the API key.
	 * @param secrets VS Code secret storage.
	 */
	constructor(
		private readonly secrets: vscode.SecretStorage,
		private readonly statusBarItem: vscode.StatusBarItem,
		reasoningState: vscode.Memento
	) {
		this._openaiChatReasoningCache = new PersistentOpenAIChatReasoningCache(reasoningState);
	}

	/**
	 * Get the list of available language models contributed by this provider
	 * @param options Options which specify the calling context of this function
	 * @param token A cancellation token which signals if the user cancelled the request or not
	 * @returns A promise that resolves to the list of available language models
	 */
	async provideLanguageModelChatInformation(
		options: { silent: boolean },
		_token: CancellationToken
	): Promise<LanguageModelChatInformation[]> {
		return prepareLanguageModelChatInformation({ silent: options.silent ?? false }, _token, this.secrets);
	}

	/**
	 * Returns the number of tokens for a given text using the model specific tokenizer logic
	 * @param model The language model to use
	 * @param text The text to count tokens for
	 * @param token A cancellation token for the request
	 * @returns A promise that resolves to the number of tokens
	 */
	async provideTokenCount(
		_model: LanguageModelChatInformation,
		text: string | LanguageModelChatRequestMessage,
		_token: CancellationToken
	): Promise<number> {
		return countMessageTokens(text, { includeReasoningInRequest: true });
	}

	/**
	 * Returns the response for a chat request, passing the results to the progress callback.
	 * The {@linkcode LanguageModelChatProvider} must emit the response parts to the progress callback as they are received from the language model.
	 * @param model The language model to use
	 * @param messages The messages to include in the request
	 * @param options Options for the request
	 * @param progress The progress to emit the streamed response chunks to
	 * @param token A cancellation token for the request
	 * @returns A promise that resolves when the response is complete. Results are actually passed to the progress callback.
	 */
	async provideLanguageModelChatResponse(
		model: LanguageModelChatInformation,
		messages: readonly LanguageModelChatRequestMessage[],
		options: ProvideLanguageModelChatResponseOptions,
		progress: Progress<LanguageModelResponsePart2>,
		token: CancellationToken
	): Promise<void> {
		const trackingProgress: Progress<LanguageModelResponsePart2> = {
			report: (part) => {
				try {
					progress.report(part);
				} catch (e) {
					console.error("[OAI Compatible Model Provider] Progress.report failed", {
						modelId: model.id,
						error: e instanceof Error ? { name: e.name, message: e.message } : String(e),
					});
				}
			},
		};
		const requestStartTime = Date.now();
		try {
			// get model config from user settings
			const config = vscode.workspace.getConfiguration();
			const userModels = normalizeUserModels(config.get<unknown>("oaicopilot.models", []));

			// Parse model ID to handle config ID
			const parsedModelId = parseModelId(model.id);

			// Find matching user model configuration
			// Prioritize matching models with same base ID and config ID
			// If no config ID, match models with same base ID
			let um: HFModelItem | undefined = userModels.find(
				(um) =>
					um.id === parsedModelId.baseId &&
					((parsedModelId.configId && um.configId === parsedModelId.configId) ||
						(!parsedModelId.configId && !um.configId))
			);

			// Only allow base-id fallback for legacy single-config models.
			// When a configId is present, falling back can incorrectly pick another variant
			// of the same base model and re-enable reasoning_content unexpectedly.
			if (!um && !parsedModelId.configId) {
				um = userModels.find((um) => um.id === parsedModelId.baseId);
			}

			// Check if using Ollama native API mode
			const apiMode = um?.apiMode ?? "openai";
			const baseUrl = um?.baseUrl || config.get<string>("oaicopilot.baseUrl", "");

			logger.info("request.start", {
				modelId: model.id,
				messageCount: messages.length,
				apiMode,
				baseUrl,
			});

			// Prepare model configuration
			const modelConfig = {
				includeReasoningInRequest: um?.include_reasoning_in_request ?? false,
			};

			// Update Token Usage
			updateContextStatusBar(messages, options.tools, model, this.statusBarItem, modelConfig);

			// Apply delay between consecutive requests
			const modelDelay = um?.delay;
			const globalDelay = config.get<number>("oaicopilot.delay", 0);
			const delayMs = modelDelay !== undefined ? modelDelay : globalDelay;

			if (delayMs > 0 && this._lastRequestTime !== null) {
				const elapsed = Date.now() - this._lastRequestTime;
				if (elapsed < delayMs) {
					const remainingDelay = delayMs - elapsed;
					logger.debug("request.delay", {
						delayMs,
						elapsed,
						remainingDelay,
					});
					await new Promise<void>((resolve) => {
						const timeout = setTimeout(() => {
							clearTimeout(timeout);
							resolve();
						}, remainingDelay);
					});
				}
			}

			// Get API key for the model's provider
			const provider = um?.owned_by;
			const useGenericKey = !um?.baseUrl;
			const modelApiKey = await this.ensureApiKey(useGenericKey, provider);
			if (!modelApiKey) {
				logger.warn("apiKey.missing", {
					provider: provider ?? "",
					useGenericKey,
				});
				throw new Error("OAI Compatible API key not found");
			}

			// send chat request
			const BASE_URL = baseUrl;
			if (!BASE_URL || !BASE_URL.startsWith("http")) {
				throw new Error(`Invalid base URL configuration.`);
			}

			// get retry config
			const retryConfig = createRetryConfig();

			// prepare headers with custom headers if specified
			const requestHeaders = CommonApi.prepareHeaders(modelApiKey, apiMode, um?.headers);
			logger.debug("request.headers", {
				headers: logger.sanitizeHeaders(requestHeaders as Record<string, string>),
			});
			logger.debug("request.messages.origin", {
				messages: messages,
			});
			if (apiMode === "ollama") {
				// Ollama native API mode
				const ollamaApi = new OllamaApi(model.id);
				const ollamaMessages = ollamaApi.convertMessages(messages, modelConfig);

				let ollamaRequestBody: OllamaRequestBody = {
					model: parsedModelId.baseId,
					messages: ollamaMessages,
					stream: true,
				};
				ollamaRequestBody = ollamaApi.prepareRequestBody(ollamaRequestBody, um, options);

				// send Ollama chat request with retry
				const url = `${BASE_URL.replace(/\/+$/, "")}/api/chat`;
				logger.debug("request.body", {
					url,
					requestBody: ollamaRequestBody,
				});
				void appendDeepSeekDebugLog("request-preflight", {
					modelId: model.id,
					baseModelId: parsedModelId.baseId,
					requestInitiator: options.requestInitiator,
					apiMode,
					baseUrl: BASE_URL,
					url,
					originalMessages: summarizeChatRequestMessages(messages),
					requestBody: ollamaRequestBody,
					debugLogPath: DEBUG_LOG_PATH,
				});
				const response = await executeWithRetry(async () => {
					const res = await fetch(url, {
						method: "POST",
						headers: requestHeaders,
						body: JSON.stringify(ollamaRequestBody),
					});

					if (!res.ok) {
						const errorText = await res.text();
						console.error("[Ollama Provider] Ollama API error response", errorText);
						void appendDeepSeekDebugLog("request-error", {
							modelId: model.id,
							baseModelId: parsedModelId.baseId,
							requestInitiator: options.requestInitiator,
							apiMode,
							baseUrl: BASE_URL,
							url,
							status: res.status,
							statusText: res.statusText,
							errorText,
							originalMessages: summarizeChatRequestMessages(messages),
							requestBody: ollamaRequestBody,
							debugLogPath: DEBUG_LOG_PATH,
						});
						throw new Error(
							`Ollama API error: [${res.status}] ${res.statusText}${errorText ? `\n${errorText}` : ""}\nURL: ${url}`
						);
					}

					return res;
				}, retryConfig);

				if (!response.body) {
					throw new Error("No response body from Ollama API");
				}
				await ollamaApi.processStreamingResponse(response.body, trackingProgress, token);
			} else if (apiMode === "anthropic") {
				// Anthropic API mode
				const anthropicApi = new AnthropicApi(model.id);
				const anthropicMessages = anthropicApi.convertMessages(messages, modelConfig);

				// requestBody
				let requestBody: AnthropicRequestBody = {
					model: parsedModelId.baseId,
					messages: anthropicMessages,
					stream: true,
				};
				requestBody = anthropicApi.prepareRequestBody(requestBody, um, options);

				// send Anthropic chat request with retry
				const normalizedBaseUrl = BASE_URL.replace(/\/+$/, "");
				// Some providers require configuring the baseUrl with a version suffix (e.g. .../v1).
				// Avoid double-appending (e.g. .../v1/v1/messages).
				const url = normalizedBaseUrl.endsWith("/v1")
					? `${normalizedBaseUrl}/messages`
					: `${normalizedBaseUrl}/v1/messages`;
				logger.debug("request.body", { url, requestBody });
				void appendDeepSeekDebugLog("request-preflight", {
					modelId: model.id,
					baseModelId: parsedModelId.baseId,
					requestInitiator: options.requestInitiator,
					apiMode,
					baseUrl: BASE_URL,
					url,
					originalMessages: summarizeChatRequestMessages(messages),
					requestBody,
					debugLogPath: DEBUG_LOG_PATH,
				});
				const response = await executeWithRetry(async () => {
					const res = await fetch(url, {
						method: "POST",
						headers: requestHeaders,
						body: JSON.stringify(requestBody),
					});

					if (!res.ok) {
						const errorText = await res.text();
						console.error("[Anthropic Provider] Anthropic API error response", errorText);
						void appendDeepSeekDebugLog("request-error", {
							modelId: model.id,
							baseModelId: parsedModelId.baseId,
							requestInitiator: options.requestInitiator,
							apiMode,
							baseUrl: BASE_URL,
							url,
							status: res.status,
							statusText: res.statusText,
							errorText,
							originalMessages: summarizeChatRequestMessages(messages),
							requestBody,
							debugLogPath: DEBUG_LOG_PATH,
						});
						throw new Error(
							`Anthropic API error: [${res.status}] ${res.statusText}${errorText ? `\n${errorText}` : ""}\nURL: ${url}`
						);
					}

					return res;
				}, retryConfig);

				if (!response.body) {
					throw new Error("No response body from Anthropic API");
				}
				await anthropicApi.processStreamingResponse(response.body, trackingProgress, token);
			} else if (apiMode === "openai-responses") {
				// OpenAI Responses API mode
				const openaiResponsesApi = new OpenaiResponsesApi(model.id);
				const normalizedBaseUrl = BASE_URL.replace(/\/+$/, "");
				const statefulModelId = parsedModelId.baseId;

				// Convert full history once (also extracts system `instructions`).
				const fullInput = openaiResponsesApi.convertMessages(messages, modelConfig);

				const marker = findLastOpenAIResponsesStatefulMarker(statefulModelId, messages);
				let deltaInput: unknown[] | null = null;
				if (marker && marker.index >= 0 && marker.index < messages.length - 1) {
					const deltaMessages = messages.slice(marker.index + 1);
					const converted = openaiResponsesApi.convertMessages(deltaMessages, modelConfig);
					if (converted.length > 0) {
						deltaInput = converted;
					}
				}

				const canUsePreviousResponseId =
					!!marker?.marker &&
					!this._openaiResponsesPreviousResponseIdUnsupportedBaseUrls.has(normalizedBaseUrl) &&
					Array.isArray(deltaInput) &&
					deltaInput.length > 0;

				const input = canUsePreviousResponseId ? deltaInput! : fullInput;

				// requestBody
				let requestBody: Record<string, unknown> = {
					model: parsedModelId.baseId,
					input,
					stream: true,
				};

				requestBody = openaiResponsesApi.prepareRequestBody(requestBody, um, options);

				// Add prompt_cache_key to enable OpenAI prompt caching.
				// Without this parameter, cached_tokens is always 0 even with identical requests.
				if (!requestBody.prompt_cache_key) {
					requestBody.prompt_cache_key = `oaicopilot-${parsedModelId.baseId}`;
				}
				// send Responses API request with retry
				const url = `${normalizedBaseUrl}/responses`;
				logger.debug("request.body", { url, requestBody });

				// If the user explicitly set `previous_response_id` via `extra`, don't apply stateful slicing.
				let addedPreviousResponseId = false;
				if (requestBody.previous_response_id !== undefined) {
					requestBody.input = fullInput;
				} else if (canUsePreviousResponseId) {
					requestBody.previous_response_id = marker!.marker;
					addedPreviousResponseId = true;
				}

				const sendRequest = async (body: Record<string, unknown>) => {
					void appendDeepSeekDebugLog("request-preflight", {
						modelId: model.id,
						baseModelId: parsedModelId.baseId,
						requestInitiator: options.requestInitiator,
						apiMode,
						baseUrl: BASE_URL,
						url,
						originalMessages: summarizeChatRequestMessages(messages),
						requestBody: body,
						debugLogPath: DEBUG_LOG_PATH,
					});
					return await executeWithRetry(async () => {
						const res = await fetch(url, {
							method: "POST",
							headers: requestHeaders,
							body: JSON.stringify(body),
						});

						if (!res.ok) {
							const errorText = await res.text();
							void appendDeepSeekDebugLog("request-error", {
								modelId: model.id,
								baseModelId: parsedModelId.baseId,
								requestInitiator: options.requestInitiator,
								apiMode,
								baseUrl: BASE_URL,
								url,
								status: res.status,
								statusText: res.statusText,
								errorText,
								originalMessages: summarizeChatRequestMessages(messages),
								requestBody: body,
								debugLogPath: DEBUG_LOG_PATH,
							});
							const error = new Error(
								`Responses API error: [${res.status}] ${res.statusText}${errorText ? `\n${errorText}` : ""}\nURL: ${url}`
							);
							(error as { status?: number; errorText?: string }).status = res.status;
							(error as { status?: number; errorText?: string }).errorText = errorText;
							throw error;
						}

						return res;
					}, retryConfig);
				};

				let response: Response;
				try {
					response = await sendRequest(requestBody);
				} catch (err) {
					// Some Responses-compatible gateways don't support `previous_response_id`.
					// Fall back to sending full history when the previous-response attempt fails.
					const status = (err as { status?: unknown })?.status;
					const shouldFallback =
						addedPreviousResponseId && typeof status === "number" && status >= 400 && status < 500 && status !== 429;
					if (!shouldFallback) {
						throw err;
					}

					this._openaiResponsesPreviousResponseIdUnsupportedBaseUrls.add(normalizedBaseUrl);

					let fallbackBody: Record<string, unknown> = {
						model: parsedModelId.baseId,
						input: fullInput,
						stream: true,
					};
					fallbackBody = openaiResponsesApi.prepareRequestBody(fallbackBody, um, options);
					delete fallbackBody.previous_response_id;
					response = await sendRequest(fallbackBody);
				}

				if (!response.body) {
					throw new Error("No response body from Responses API");
				}
				await openaiResponsesApi.processStreamingResponse(response.body, trackingProgress, token);

				// Append a stateful marker so future requests can reuse `previous_response_id` (Copilot Chat style).
				const responseId = openaiResponsesApi.responseId;
				if (responseId) {
					trackingProgress.report(createOpenAIResponsesStatefulMarkerPart(statefulModelId, responseId));
				}
			} else if (apiMode === "gemini") {
				// Gemini native API mode
				const geminiApi = new GeminiApi(model.id, this._geminiToolCallMetaByCallId);
				const geminiMessages = geminiApi.convertMessages(messages, modelConfig);

				const systemParts: string[] = [];
				const contents: GeminiGenerateContentRequest["contents"] = [];
				for (const msg of geminiMessages) {
					if (msg.role === "system") {
						const text = msg.parts
							.map((p) =>
								p && typeof p === "object" && typeof (p as { text?: unknown }).text === "string"
									? String((p as { text: string }).text)
									: ""
							)
							.join("")
							.trim();
						if (text) {
							systemParts.push(text);
						}
						continue;
					}
					contents.push({ role: msg.role, parts: msg.parts });
				}

				let requestBody: GeminiGenerateContentRequest = {
					contents,
				};
				if (systemParts.length > 0) {
					requestBody.systemInstruction = { role: "user", parts: [{ text: systemParts.join("\n") }] };
				}
				requestBody = geminiApi.prepareRequestBody(requestBody, um, options);

				const url = buildGeminiGenerateContentUrl(BASE_URL, parsedModelId.baseId, true);
				logger.debug("request.body", { url, requestBody });
				if (!url) {
					throw new Error("Invalid Gemini base URL configuration.");
				}

				void appendDeepSeekDebugLog("request-preflight", {
					modelId: model.id,
					baseModelId: parsedModelId.baseId,
					requestInitiator: options.requestInitiator,
					apiMode,
					baseUrl: BASE_URL,
					url,
					originalMessages: summarizeChatRequestMessages(messages),
					requestBody,
					debugLogPath: DEBUG_LOG_PATH,
				});
				const response = await executeWithRetry(async () => {
					const res = await fetch(url, {
						method: "POST",
						headers: requestHeaders,
						body: JSON.stringify(requestBody),
					});

					if (!res.ok) {
						const errorText = await res.text();
						console.error("[Gemini Provider] Gemini API error response", errorText);
						void appendDeepSeekDebugLog("request-error", {
							modelId: model.id,
							baseModelId: parsedModelId.baseId,
							requestInitiator: options.requestInitiator,
							apiMode,
							baseUrl: BASE_URL,
							url,
							status: res.status,
							statusText: res.statusText,
							errorText,
							originalMessages: summarizeChatRequestMessages(messages),
							requestBody,
							debugLogPath: DEBUG_LOG_PATH,
						});
						throw new Error(
							`Gemini API error: [${res.status}] ${res.statusText}${errorText ? `\n${errorText}` : ""}\nURL: ${url}`
						);
					}

					return res;
				}, retryConfig);

				if (!response.body) {
					throw new Error("No response body from Gemini API");
				}
				await geminiApi.processStreamingResponse(response.body, trackingProgress, token);
			} else {
				// OpenAI compatible API mode (default)
				const openaiApi = new OpenaiApi(model.id);
				const reasoningModelScopeId = parsedModelId.configId ? model.id : parsedModelId.baseId;
				const openAIChatConversationState = modelConfig.includeReasoningInRequest
					? getOrCreateOpenAIChatConversationState(reasoningModelScopeId, options.requestInitiator, messages)
					: null;
				const restoredMessages = modelConfig.includeReasoningInRequest
					? restoreOpenAIChatReasoningMessages(
							reasoningModelScopeId,
							options.requestInitiator,
							messages,
							this._openaiChatReasoningCache
						)
					: messages;
				const openaiMessages = openaiApi.convertMessages(restoredMessages, modelConfig);

				// requestBody
				let requestBody: Record<string, unknown> = {
					model: parsedModelId.baseId,
					messages: openaiMessages,
					stream: true,
					stream_options: { include_usage: true },
				};
				requestBody = openaiApi.prepareRequestBody(requestBody, um, options);

				// send chat request with retry
				const url = `${BASE_URL.replace(/\/+$/, "")}/chat/completions`;
				logger.debug("request.body", { url, requestBody });
				void appendDeepSeekDebugLog("request-preflight", {
					modelId: model.id,
					baseModelId: parsedModelId.baseId,
					requestInitiator: options.requestInitiator,
					apiMode,
					baseUrl: BASE_URL,
					url,
					includeReasoningInRequest: modelConfig.includeReasoningInRequest,
					reasoningModelScopeId,
					conversationId: openAIChatConversationState?.conversationId ?? null,
					originalMessages: summarizeChatRequestMessages(messages),
					restoredMessages: summarizeChatRequestMessages(restoredMessages),
					openaiMessages: summarizeOpenAIChatMessages(openaiMessages),
					requestBody,
					debugLogPath: DEBUG_LOG_PATH,
				});
				const response = await executeWithRetry(async () => {
					const res = await fetch(url, {
						method: "POST",
						headers: requestHeaders,
						body: JSON.stringify(requestBody),
					});

						if (!res.ok) {
							const errorText = await res.text();
							console.error("[OAI Compatible Model Provider] OAI Compatible API error response", errorText);
							{
								void appendDeepSeekDebugLog("request-error", {
									modelId: model.id,
									baseModelId: parsedModelId.baseId,
									requestInitiator: options.requestInitiator,
									apiMode,
									baseUrl: BASE_URL,
									url,
									status: res.status,
									statusText: res.statusText,
									errorText,
									includeReasoningInRequest: modelConfig.includeReasoningInRequest,
									reasoningModelScopeId,
									conversationId: openAIChatConversationState?.conversationId ?? null,
									originalMessages: summarizeChatRequestMessages(messages),
									restoredMessages: summarizeChatRequestMessages(restoredMessages),
									openaiMessages: summarizeOpenAIChatMessages(openaiMessages),
									requestBody,
									debugLogPath: DEBUG_LOG_PATH,
								});
							}
							throw new Error(
								`OAI Compatible API error: [${res.status}] ${res.statusText}${errorText ? `\n${errorText}` : ""}\nURL: ${url}`
							);
						}

					return res;
				}, retryConfig);

				if (!response.body) {
					throw new Error("No response body from OAI Compatible API");
				}

				let aggregatedReasoning = "";
				let aggregatedAssistantText = "";
				const responseToolCalls: OpenAIChatToolCallSignature[] = [];
				const openaiTrackingProgress: Progress<LanguageModelResponsePart2> = {
					report: (part) => {
						if (part instanceof vscode.LanguageModelThinkingPart) {
							const text = Array.isArray(part.value) ? part.value.join("") : part.value;
							if (text) {
								aggregatedReasoning += text;
							}
						} else if (part instanceof vscode.LanguageModelTextPart) {
							aggregatedAssistantText += part.value;
						} else if (part instanceof vscode.LanguageModelToolCallPart) {
							responseToolCalls.push(createOpenAIChatToolCallSignature(part));
						}
						trackingProgress.report(part);
					},
				};

				await openaiApi.processStreamingResponse(response.body, openaiTrackingProgress, token);

				const hasVisibleOrReasoningResponse =
					!!aggregatedAssistantText || responseToolCalls.length > 0 || !!aggregatedReasoning.trim();
				if (modelConfig.includeReasoningInRequest && openAIChatConversationState && hasVisibleOrReasoningResponse) {
					const messageId = randomUUID();
					trackingProgress.report(
						createOpenAIChatStateMarkerPart(
							reasoningModelScopeId,
							options.requestInitiator,
							openAIChatConversationState.conversationId,
							messageId
						)
					);

					if (aggregatedReasoning.trim()) {
						trackingProgress.report(
							createOpenAIChatReasoningMarkerPart(
								reasoningModelScopeId,
								options.requestInitiator,
								openAIChatConversationState.conversationId,
								messageId,
								aggregatedReasoning
							)
						);
						await this._openaiChatReasoningCache.remember(
							reasoningModelScopeId,
							options.requestInitiator,
							openAIChatConversationState.conversationId,
							messageId,
							aggregatedReasoning,
							messages,
							aggregatedAssistantText,
							responseToolCalls
						);
					}
				}
			}
		} catch (err) {
			console.error("[OAI Compatible Model Provider] Chat request failed", {
				modelId: model.id,
				messageCount: messages.length,
				error: err instanceof Error ? { name: err.name, message: err.message } : String(err),
			});
			logger.error("request.error", {
				modelId: model.id,
				messageCount: messages.length,
				errorName: err instanceof Error ? err.name : String(err),
				errorMessage: err instanceof Error ? err.message : String(err),
			});
			throw err;
		} finally {
			const durationMs = Date.now() - requestStartTime;
			logger.info("request.end", { modelId: model.id, durationMs });
			// Update last request time after successful completion
			this._lastRequestTime = Date.now();
		}
	}

	/**
	 * Ensure an API key exists in SecretStorage, optionally prompting the user when not silent.
	 * @param useGenericKey If true, use generic API key.
	 * @param provider Optional provider name to get provider-specific API key.
	 */
	private async ensureApiKey(useGenericKey: boolean, provider?: string): Promise<string | undefined> {
		// Try to get provider-specific API key first
		let apiKey: string | undefined;
		if (provider && provider.trim() !== "") {
			const normalizedProvider = provider.trim().toLowerCase();
			const providerKey = `oaicopilot.apiKey.${normalizedProvider}`;
			apiKey = await this.secrets.get(providerKey);

			if (!apiKey && !useGenericKey) {
				const entered = await vscode.window.showInputBox({
					title: `OAI Compatible API Key for ${normalizedProvider}`,
					prompt: `Enter your OAI Compatible API key for ${normalizedProvider}`,
					ignoreFocusOut: true,
					password: true,
				});
				if (entered && entered.trim()) {
					apiKey = entered.trim();
					await this.secrets.store(providerKey, apiKey);
				}
			}
		}

		// Fall back to generic API key
		if (!apiKey) {
			apiKey = await this.secrets.get("oaicopilot.apiKey");
		}

		if (!apiKey && useGenericKey) {
			const entered = await vscode.window.showInputBox({
				title: "OAI Compatible API Key",
				prompt: "Enter your OAI Compatible API key",
				ignoreFocusOut: true,
				password: true,
			});
			if (entered && entered.trim()) {
				apiKey = entered.trim();
				await this.secrets.store("oaicopilot.apiKey", apiKey);
			}
		}
		return apiKey;
	}
}

type OpenAIResponsesStatefulMarkerLocation = { marker: string; index: number };
type OpenAIChatToolCallSignature = { callId: string; name: string; inputJson: string };

type OpenAIChatConversationState = { conversationId: string };
type OpenAIChatStateMarkerPayload = {
	version: 1;
	modelId: string;
	requestInitiator?: string;
	conversationId: string;
	messageId: string;
};
type OpenAIChatReasoningMarkerPayload = OpenAIChatStateMarkerPayload & {
	reasoning: string;
};

const OPENAI_CHAT_REASONING_CACHE_TOTAL_LIMIT = 8192;
const OPENAI_CHAT_REASONING_CACHE_SESSION_LIMIT = 2048;
const OPENAI_CHAT_REASONING_CACHE_TTL_MS = 1000 * 60 * 60 * 24 * 30;

function createOpenAIResponsesStatefulMarkerPart(modelId: string, marker: string): vscode.LanguageModelDataPart {
	const payload = `${modelId}\\${marker}`;
	const bytes = new TextEncoder().encode(payload);
	return new vscode.LanguageModelDataPart(bytes, HuggingFaceChatModelProvider.OPENAI_RESPONSES_STATEFUL_MARKER_MIME);
}

function parseOpenAIResponsesStatefulMarkerPart(part: unknown): { modelId: string; marker: string } | null {
	const maybe = part as { mimeType?: unknown; data?: unknown };
	if (!maybe || typeof maybe !== "object") {
		return null;
	}
	if (typeof maybe.mimeType !== "string") {
		return null;
	}
	if (!(maybe.data instanceof Uint8Array)) {
		return null;
	}
	if (maybe.mimeType !== HuggingFaceChatModelProvider.OPENAI_RESPONSES_STATEFUL_MARKER_MIME) {
		return null;
	}

	try {
		const decoded = new TextDecoder().decode(maybe.data);
		const sep = decoded.indexOf("\\");
		if (sep <= 0) {
			return null;
		}
		const modelId = decoded.slice(0, sep).trim();
		const marker = decoded.slice(sep + 1).trim();
		if (!modelId || !marker) {
			return null;
		}
		return { modelId, marker };
	} catch {
		return null;
	}
}

function createCompressedMarkerPart(mimeType: string, payload: Record<string, unknown>): vscode.LanguageModelDataPart {
	return new vscode.LanguageModelDataPart(gzipSync(JSON.stringify(payload)), mimeType);
}

function parseCompressedMarkerPart<T extends Record<string, unknown>>(part: unknown, mimeType: string): T | null {
	const maybe = part as { mimeType?: unknown; data?: unknown };
	if (!maybe || typeof maybe !== "object") {
		return null;
	}
	if (maybe.mimeType !== mimeType) {
		return null;
	}
	if (!(maybe.data instanceof Uint8Array)) {
		return null;
	}

	try {
		const decoded = gunzipSync(Buffer.from(maybe.data)).toString("utf8");
		const parsed = JSON.parse(decoded);
		return parsed && typeof parsed === "object" ? (parsed as T) : null;
	} catch {
		return null;
	}
}

function createOpenAIChatStateMarkerPart(
	modelId: string,
	requestInitiator: string,
	conversationId: string,
	messageId: string
): vscode.LanguageModelDataPart {
	return createCompressedMarkerPart(HuggingFaceChatModelProvider.OPENAI_CHAT_STATE_MARKER_MIME, {
		version: 1,
		modelId,
		requestInitiator,
		conversationId,
		messageId,
	});
}

function parseOpenAIChatStateMarkerPart(part: unknown): OpenAIChatStateMarkerPayload | null {
	const parsed = parseCompressedMarkerPart<OpenAIChatStateMarkerPayload>(
		part,
		HuggingFaceChatModelProvider.OPENAI_CHAT_STATE_MARKER_MIME
	);
	if (!parsed || parsed.version !== 1) {
		return null;
	}
	if (!parsed.modelId || !parsed.conversationId || !parsed.messageId) {
		return null;
	}
	return parsed;
}

function createOpenAIChatReasoningMarkerPart(
	modelId: string,
	requestInitiator: string,
	conversationId: string,
	messageId: string,
	reasoning: string
): vscode.LanguageModelDataPart {
	return createCompressedMarkerPart(HuggingFaceChatModelProvider.OPENAI_CHAT_REASONING_MARKER_MIME, {
		version: 1,
		modelId,
		requestInitiator,
		conversationId,
		messageId,
		reasoning,
	});
}

function markerMatchesRequestInitiator(
	marker: { requestInitiator?: string },
	requestInitiator: string,
	allowLegacyMarker: boolean
): boolean {
	if (!marker.requestInitiator) {
		return allowLegacyMarker;
	}
	return marker.requestInitiator === requestInitiator;
}

function parseOpenAIChatReasoningMarkerPart(part: unknown): OpenAIChatReasoningMarkerPayload | null {
	const parsed = parseCompressedMarkerPart<OpenAIChatReasoningMarkerPayload>(
		part,
		HuggingFaceChatModelProvider.OPENAI_CHAT_REASONING_MARKER_MIME
	);
	if (!parsed || parsed.version !== 1) {
		return null;
	}
	if (!parsed.modelId || !parsed.conversationId || !parsed.messageId || typeof parsed.reasoning !== "string") {
		return null;
	}
	return parsed;
}

function extractOpenAIChatMessageState(
	modelId: string,
	content: readonly unknown[],
	requestInitiator: string,
	allowLegacyMarker = true
): OpenAIChatStateMarkerPayload | null {
	for (const part of content) {
		const stateMarker = parseOpenAIChatStateMarkerPart(part);
		if (
			stateMarker &&
			stateMarker.modelId === modelId &&
			markerMatchesRequestInitiator(stateMarker, requestInitiator, allowLegacyMarker)
		) {
			return stateMarker;
		}
		const reasoningMarker = parseOpenAIChatReasoningMarkerPart(part);
		if (
			reasoningMarker &&
			reasoningMarker.modelId === modelId &&
			markerMatchesRequestInitiator(reasoningMarker, requestInitiator, allowLegacyMarker)
		) {
			return reasoningMarker;
		}
	}
	return null;
}

function extractOpenAIChatMessageReasoning(
	modelId: string,
	content: readonly unknown[],
	requestInitiator: string
): OpenAIChatReasoningMarkerPayload | null {
	for (const part of content) {
		const marker = parseOpenAIChatReasoningMarkerPart(part);
		if (
			marker &&
			marker.modelId === modelId &&
			markerMatchesRequestInitiator(marker, requestInitiator, false)
		) {
			return marker;
		}
	}
	return null;
}

function getOrCreateOpenAIChatConversationState(
	modelId: string,
	requestInitiator: string,
	messages: readonly LanguageModelChatRequestMessage[]
): OpenAIChatConversationState {
	for (let i = messages.length - 1; i >= 0; i--) {
		const message = messages[i];
		if (message.role !== vscode.LanguageModelChatMessageRole.Assistant) {
			continue;
		}
		const state = extractOpenAIChatMessageState(modelId, message.content ?? [], requestInitiator, true);
		if (state) {
			return { conversationId: state.conversationId };
		}
	}
	return { conversationId: randomUUID() };
}

function findLastOpenAIResponsesStatefulMarker(
	modelId: string,
	messages: readonly LanguageModelChatRequestMessage[]
): OpenAIResponsesStatefulMarkerLocation | null {
	for (let i = messages.length - 1; i >= 0; i--) {
		if (messages[i].role !== vscode.LanguageModelChatMessageRole.Assistant) {
			continue;
		}
		for (const part of messages[i].content ?? []) {
			const parsed = parseOpenAIResponsesStatefulMarkerPart(part);
			if (parsed && parsed.modelId === modelId) {
				return { marker: parsed.marker, index: i };
			}
		}
	}
	return null;
}

function restoreOpenAIChatReasoningMessages(
	modelId: string,
	requestInitiator: string,
	messages: readonly LanguageModelChatRequestMessage[],
	reasoningCache: PersistentOpenAIChatReasoningCache
): readonly LanguageModelChatRequestMessage[] {
	return messages.map((message, messageIndex) => {
		if (message.role !== vscode.LanguageModelChatMessageRole.Assistant) {
			return message;
		}

		const content = [...(message.content ?? [])];
		const hasThinking = content.some((part) => {
			if (!(part instanceof vscode.LanguageModelThinkingPart)) {
				return false;
			}
			const value = Array.isArray(part.value) ? part.value.join("") : part.value;
			return !!value.trim();
		});
		if (hasThinking) {
			return message;
		}

		const markerReasoning = extractOpenAIChatMessageReasoning(modelId, content, requestInitiator)?.reasoning ?? "";
		const stateMarker = extractOpenAIChatMessageState(modelId, content, requestInitiator, true);
		let restoredReasoning = markerReasoning;
		if (!restoredReasoning && stateMarker) {
			restoredReasoning = reasoningCache.recall(
				modelId,
				requestInitiator,
				stateMarker.conversationId,
				stateMarker.messageId
			);
		}
		if (!restoredReasoning) {
			const { assistantText, toolCalls } = extractAssistantMessageSignature(content);
			// VS Code can omit internal marker/data parts when replaying assistant turns back
			// into the provider. Fall back to the persisted signature-based lookup so
			// preserved thinking still works for the same request initiator.
			restoredReasoning = reasoningCache.recallLegacy(
				modelId,
				requestInitiator,
				messages.slice(0, messageIndex),
				assistantText,
				toolCalls
			);
		}
		if (!restoredReasoning) {
			return message;
		}

		return {
			...message,
			content: [...content, new vscode.LanguageModelThinkingPart(restoredReasoning)],
		} as LanguageModelChatRequestMessage;
	});
}

function createOpenAIChatToolCallSignature(part: vscode.LanguageModelToolCallPart): OpenAIChatToolCallSignature {
	return {
		callId: part.callId || "",
		name: part.name,
		inputJson: stableStringify(part.input ?? {}),
	};
}

function extractAssistantMessageSignature(content: readonly unknown[]): {
	assistantText: string;
	toolCalls: OpenAIChatToolCallSignature[];
} {
	let assistantText = "";
	const toolCalls: OpenAIChatToolCallSignature[] = [];
	for (const part of content) {
		if (part instanceof vscode.LanguageModelTextPart) {
			assistantText += part.value;
		} else if (part instanceof vscode.LanguageModelToolCallPart) {
			toolCalls.push(createOpenAIChatToolCallSignature(part));
		}
	}
	return { assistantText, toolCalls };
}

function extractOpenAIChatReasoningAnchor(
	prefixMessages: readonly LanguageModelChatRequestMessage[]
): Record<string, unknown> | null {
	for (let index = prefixMessages.length - 1; index >= 0; index--) {
		const message = prefixMessages[index];
		if (message.role === vscode.LanguageModelChatMessageRole.Assistant) {
			continue;
		}
		const normalized = normalizeMessageForReasoningCache(message);
		const normalizedContent = normalized.content;
		if (Array.isArray(normalizedContent) && normalizedContent.length > 0) {
			return normalized;
		}
	}
	return null;
}

function buildOpenAIChatReasoningTurnCacheKey(
	modelId: string,
	requestInitiator: string,
	prefixMessages: readonly LanguageModelChatRequestMessage[],
	assistantText: string,
	toolCalls: readonly OpenAIChatToolCallSignature[]
): string {
	return stableStringify({
		modelId,
		requestInitiator,
		recentAnchor: extractOpenAIChatReasoningAnchor(prefixMessages),
		assistantText,
		toolCalls,
	});
}

function buildOpenAIChatReasoningFallbackCacheKey(
	modelId: string,
	requestInitiator: string,
	assistantText: string,
	toolCalls: readonly OpenAIChatToolCallSignature[]
): string {
	return stableStringify({
		modelId,
		requestInitiator,
		assistantText,
		toolCalls,
	});
}

function normalizeMessageForReasoningCache(message: LanguageModelChatRequestMessage): Record<string, unknown> {
	const normalizedContent: Array<Record<string, unknown>> = [];
	for (const part of message.content ?? []) {
		if (part instanceof vscode.LanguageModelTextPart) {
			if (part.value) {
				normalizedContent.push({ type: "text", value: part.value });
			}
			continue;
		}
		if (part instanceof vscode.LanguageModelToolCallPart) {
			normalizedContent.push({ type: "tool_call", ...createOpenAIChatToolCallSignature(part) });
			continue;
		}
		if (isToolResultPart(part)) {
			normalizedContent.push({
				type: "tool_result",
				callId: part.callId,
				content: collectToolResultText(part),
			});
			continue;
		}
		if (part instanceof vscode.LanguageModelDataPart) {
			if (isInternalMarkerMimeType(part.mimeType)) {
				continue;
			}
			normalizedContent.push({ type: "data", mimeType: part.mimeType, byteLength: part.data.byteLength });
		}
	}
	return {
		role: mapRole(message),
		content: normalizedContent,
	};
}

function stableStringify(value: unknown): string {
	if (Array.isArray(value)) {
		return `[${value.map((item) => stableStringify(item)).join(",")}]`;
	}
	if (value && typeof value === "object") {
		const entries = Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left.localeCompare(right));
		return `{${entries.map(([key, entryValue]) => `${JSON.stringify(key)}:${stableStringify(entryValue)}`).join(",")}}`;
	}
	return JSON.stringify(value);
}

type PersistentReasoningEntry = {
	key: string;
	payload: string;
	updatedAt: number;
};

type PersistentReasoningMessageRecord = {
	messageId: string;
	payload: string;
	updatedAt: number;
};

type PersistentReasoningSessionRecord = {
	sessionKey: string;
	modelId: string;
	requestInitiator: string;
	conversationId: string;
	updatedAt: number;
	messages: PersistentReasoningMessageRecord[];
};

type PersistentReasoningSessionBucket = Omit<PersistentReasoningSessionRecord, "messages"> & {
	messages: Map<string, PersistentReasoningMessageRecord>;
};

type PersistentReasoningIndexRecord = {
	key: string;
	sessionKey: string;
	messageId: string;
	updatedAt: number;
};

class PersistentOpenAIChatReasoningCache {
	private static readonly LEGACY_STORAGE_KEY = "oaicopilot.openaiChatReasoningCache.v2";
	private static readonly STORAGE_KEY = "oaicopilot.openaiChatReasoningCache.v3";
	private static readonly INDEX_STORAGE_KEY = "oaicopilot.openaiChatReasoningCache.v3.indices";
	private readonly sessions = new Map<string, PersistentReasoningSessionBucket>();
	private readonly legacyEntries = new Map<string, PersistentReasoningEntry>();
	private readonly indices = new Map<string, PersistentReasoningIndexRecord>();

	constructor(private readonly state: vscode.Memento) {
		const now = Date.now();
		const stored = state.get<PersistentReasoningSessionRecord[]>(PersistentOpenAIChatReasoningCache.STORAGE_KEY, []);
		for (const session of stored) {
			if (!session?.sessionKey || !session.modelId || !session.requestInitiator || !session.conversationId) {
				continue;
			}
			if (typeof session.updatedAt !== "number" || now - session.updatedAt > OPENAI_CHAT_REASONING_CACHE_TTL_MS) {
				continue;
			}
			const messages = new Map<string, PersistentReasoningMessageRecord>();
			for (const entry of session.messages ?? []) {
				if (!entry?.messageId || !entry.payload || typeof entry.updatedAt !== "number") {
					continue;
				}
				if (now - entry.updatedAt > OPENAI_CHAT_REASONING_CACHE_TTL_MS) {
					continue;
				}
				messages.set(entry.messageId, entry);
			}
			if (messages.size === 0) {
				continue;
			}
			this.sessions.set(session.sessionKey, { ...session, messages });
		}

		const legacy = state.get<PersistentReasoningEntry[]>(PersistentOpenAIChatReasoningCache.LEGACY_STORAGE_KEY, []);
		for (const entry of legacy) {
			if (entry?.key && entry?.payload) {
				this.legacyEntries.set(entry.key, entry);
			}
		}

		const indices = state.get<PersistentReasoningIndexRecord[]>(PersistentOpenAIChatReasoningCache.INDEX_STORAGE_KEY, []);
		for (const entry of indices) {
			if (!entry?.key || !entry.sessionKey || !entry.messageId || typeof entry.updatedAt !== "number") {
				continue;
			}
			if (now - entry.updatedAt > OPENAI_CHAT_REASONING_CACHE_TTL_MS) {
				continue;
			}
			this.indices.set(entry.key, entry);
		}
		this.pruneInMemory();
	}

	recall(
		modelId: string,
		requestInitiator: string,
		conversationId: string,
		messageId: string
	): string {
		const sessionKey = this.buildSessionKey(modelId, requestInitiator, conversationId);
		const entry = this.touchSessionMessage(sessionKey, messageId);
		if (!entry) {
			return "";
		}
		try {
			return this.decodePayload(entry.payload);
		} catch {
			this.removeMessage(sessionKey, messageId);
			void this.flush();
			return "";
		}
	}

	recallLegacy(
		modelId: string,
		requestInitiator: string,
		prefixMessages: readonly LanguageModelChatRequestMessage[],
		assistantText: string,
		toolCalls: readonly OpenAIChatToolCallSignature[]
	): string {
		const key = buildOpenAIChatReasoningTurnCacheKey(
			modelId,
			requestInitiator,
			prefixMessages,
			assistantText,
			toolCalls
		);
		const fallbackKey = buildOpenAIChatReasoningFallbackCacheKey(
			modelId,
			requestInitiator,
			assistantText,
			toolCalls
		);
		const indexed = this.indices.get(key) ?? this.indices.get(fallbackKey);
		if (indexed) {
			const entry = this.touchSessionMessage(indexed.sessionKey, indexed.messageId);
			if (entry) {
				try {
					return this.decodePayload(entry.payload);
				} catch {
					this.removeMessage(indexed.sessionKey, indexed.messageId);
					void this.flush();
					return "";
				}
			}
			this.indices.delete(indexed.key);
			void this.flush();
		}
		const matchedKey = this.legacyEntries.has(key) ? key : this.legacyEntries.has(fallbackKey) ? fallbackKey : "";
		const entry = matchedKey ? this.legacyEntries.get(matchedKey) : undefined;
		if (!entry) {
			return "";
		}
		try {
			return this.decodePayload(entry.payload);
		} catch {
			this.legacyEntries.delete(matchedKey);
			return "";
		}
	}

	async remember(
		modelId: string,
		requestInitiator: string,
		conversationId: string,
		messageId: string,
		reasoning: string,
		prefixMessages?: readonly LanguageModelChatRequestMessage[],
		assistantText?: string,
		toolCalls?: readonly OpenAIChatToolCallSignature[]
	): Promise<void> {
		const sessionKey = this.buildSessionKey(modelId, requestInitiator, conversationId);
		const now = Date.now();
		const payload = gzipSync(reasoning).toString("base64url");
		let session = this.sessions.get(sessionKey);
		if (!session) {
			session = {
				sessionKey,
				modelId,
				requestInitiator,
				conversationId,
				updatedAt: now,
				messages: new Map<string, PersistentReasoningMessageRecord>(),
			};
			this.sessions.set(sessionKey, session);
		}
		session.updatedAt = now;
		if (session.messages.has(messageId)) {
			session.messages.delete(messageId);
		}
		session.messages.set(messageId, { messageId, payload, updatedAt: now });
		if (prefixMessages && assistantText !== undefined && toolCalls) {
			const keys = new Set([
				buildOpenAIChatReasoningTurnCacheKey(modelId, requestInitiator, prefixMessages, assistantText, toolCalls),
				buildOpenAIChatReasoningFallbackCacheKey(modelId, requestInitiator, assistantText, toolCalls),
			]);
			for (const key of keys) {
				this.indices.set(key, { key, sessionKey, messageId, updatedAt: now });
			}
		}
		this.pruneInMemory();
		await this.flush();
	}

	private buildSessionKey(modelId: string, requestInitiator: string, conversationId: string): string {
		return stableStringify({ modelId, requestInitiator, conversationId });
	}

	private touchSessionMessage(sessionKey: string, messageId: string): PersistentReasoningMessageRecord | null {
		const session = this.sessions.get(sessionKey);
		if (!session) {
			return null;
		}
		const entry = session.messages.get(messageId);
		if (!entry) {
			return null;
		}
		const updatedAt = Date.now();
		const nextEntry = { ...entry, updatedAt };
		session.messages.delete(messageId);
		session.messages.set(messageId, nextEntry);
		session.updatedAt = updatedAt;
		for (const index of this.indices.values()) {
			if (index.sessionKey === sessionKey && index.messageId === messageId) {
				index.updatedAt = updatedAt;
			}
		}
		return nextEntry;
	}

	private removeMessage(sessionKey: string, messageId: string): void {
		const session = this.sessions.get(sessionKey);
		if (session) {
			session.messages.delete(messageId);
			if (session.messages.size === 0) {
				this.sessions.delete(sessionKey);
			}
		}
		for (const [key, index] of this.indices) {
			if (index.sessionKey === sessionKey && index.messageId === messageId) {
				this.indices.delete(key);
			}
		}
	}

	private removeSession(sessionKey: string): void {
		this.sessions.delete(sessionKey);
		for (const [key, index] of this.indices) {
			if (index.sessionKey === sessionKey) {
				this.indices.delete(key);
			}
		}
	}

	private decodePayload(payload: string): string {
		return gunzipSync(Buffer.from(payload, "base64url")).toString("utf8");
	}

	private pruneInMemory(): void {
		const now = Date.now();
		for (const [sessionKey, session] of this.sessions) {
			if (now - session.updatedAt > OPENAI_CHAT_REASONING_CACHE_TTL_MS) {
				this.removeSession(sessionKey);
				continue;
			}
			for (const [messageId, entry] of session.messages) {
				if (now - entry.updatedAt > OPENAI_CHAT_REASONING_CACHE_TTL_MS) {
					this.removeMessage(sessionKey, messageId);
				}
			}
			while (session.messages.size > OPENAI_CHAT_REASONING_CACHE_SESSION_LIMIT) {
				const oldestMessageId = session.messages.keys().next().value;
				if (!oldestMessageId) {
					break;
				}
				this.removeMessage(sessionKey, oldestMessageId);
			}
			if (session.messages.size === 0) {
				this.removeSession(sessionKey);
			}
		}

		for (const [key, index] of this.indices) {
			if (now - index.updatedAt > OPENAI_CHAT_REASONING_CACHE_TTL_MS) {
				this.indices.delete(key);
				continue;
			}
			const session = this.sessions.get(index.sessionKey);
			if (!session || !session.messages.has(index.messageId)) {
				this.indices.delete(key);
			}
		}

		while (this.totalMessageCount() > OPENAI_CHAT_REASONING_CACHE_TOTAL_LIMIT) {
			const oldest = this.findOldestMessage();
			if (!oldest) {
				break;
			}
			const session = this.sessions.get(oldest.sessionKey);
			if (!session) {
				break;
			}
			this.removeMessage(oldest.sessionKey, oldest.messageId);
		}
	}

	private totalMessageCount(): number {
		let total = 0;
		for (const session of this.sessions.values()) {
			total += session.messages.size;
		}
		return total;
	}

	private findOldestMessage(): { sessionKey: string; messageId: string; updatedAt: number } | null {
		let oldest: { sessionKey: string; messageId: string; updatedAt: number } | null = null;
		for (const [sessionKey, session] of this.sessions) {
			for (const [messageId, entry] of session.messages) {
				if (!oldest || entry.updatedAt < oldest.updatedAt) {
					oldest = { sessionKey, messageId, updatedAt: entry.updatedAt };
				}
			}
		}
		return oldest;
	}

	private async flush(): Promise<void> {
		await this.state.update(
			PersistentOpenAIChatReasoningCache.STORAGE_KEY,
			Array.from(this.sessions.values()).map((session) => ({
				sessionKey: session.sessionKey,
				modelId: session.modelId,
				requestInitiator: session.requestInitiator,
				conversationId: session.conversationId,
				updatedAt: session.updatedAt,
				messages: Array.from(session.messages.values()),
			}))
		);
		await this.state.update(PersistentOpenAIChatReasoningCache.INDEX_STORAGE_KEY, Array.from(this.indices.values()));
	}
}
