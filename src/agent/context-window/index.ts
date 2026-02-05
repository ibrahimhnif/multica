/**
 * Context Window Guard
 *
 * 提供上下文窗口管理功能，防止 token 溢出
 */

// Types
export type {
  ContextWindowSource,
  ContextWindowInfo,
  ContextWindowGuardResult,
  TokenEstimation,
  TokenAwareCompactionResult,
} from "./types.js";

// Guard
export {
  CONTEXT_WINDOW_HARD_MIN_TOKENS,
  CONTEXT_WINDOW_WARN_BELOW_TOKENS,
  DEFAULT_CONTEXT_TOKENS,
  resolveContextWindowInfo,
  evaluateContextWindowGuard,
  checkContextWindow,
} from "./guard.js";

// Token estimation
export {
  ESTIMATION_SAFETY_MARGIN,
  COMPACTION_TRIGGER_RATIO,
  COMPACTION_TARGET_RATIO,
  MIN_KEEP_MESSAGES,
  estimateMessagesTokens,
  estimateSystemPromptTokens,
  estimateTokenUsage,
  shouldCompact,
  compactMessagesTokenAware,
  isMessageOversized,
} from "./token-estimation.js";

// Summarization
export type { SummaryCompactionResult, SummaryCompactionParams } from "./summarization.js";
export {
  splitMessagesForSummary,
  compactMessagesWithSummary,
  compactMessagesWithChunkedSummary,
} from "./summarization.js";

// Tool result pruning
export type {
  ToolResultPruningSettings,
  ToolResultPruningResult,
} from "./tool-result-pruning.js";
export {
  DEFAULT_TOOL_RESULT_PRUNING_SETTINGS,
  pruneToolResults,
} from "./tool-result-pruning.js";
