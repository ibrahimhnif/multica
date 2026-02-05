import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { Model } from "@mariozechner/pi-ai";
import {
  estimateMessagesTokens,
  compactMessagesTokenAware,
  estimateTokenUsage,
  shouldCompact as shouldCompactTokens,
  compactMessagesWithSummary,
  compactMessagesWithChunkedSummary,
  COMPACTION_TARGET_RATIO,
  MIN_KEEP_MESSAGES,
} from "../context-window/index.js";

export type CompactionResult = {
  kept: AgentMessage[];
  removedCount: number;
  /** Additional information in token-aware mode */
  tokensRemoved?: number | undefined;
  tokensKept?: number | undefined;
  /** Summary generated in summary mode */
  summary?: string | undefined;
  /** Reason for compaction: count, tokens, summary, or pruning (tool result trimming only) */
  reason: "count" | "tokens" | "summary" | "pruning";
};

/**
 * Find a safe compaction point that doesn't break tool_use/tool_result pairs.
 * Returns the index to start keeping messages from.
 */
function findSafeCompactionPoint(messages: AgentMessage[], targetStart: number): number {
  let start = targetStart;

  // Move forward until we find a safe starting point
  while (start < messages.length) {
    const msg = messages[start];
    if (!msg) {
      start++;
      continue;
    }

    // Safe to start from a user message
    if (msg.role === "user") {
      // But make sure it's not a toolResult without corresponding tool_use
      const msgAny = msg as any;
      if (Array.isArray(msgAny.content)) {
        const hasToolResult = msgAny.content.some((b: any) => b.type === "tool_result");
        if (!hasToolResult) {
          break; // Safe: user message without tool_result
        }
      } else {
        break; // Safe: simple user message
      }
    }

    // toolResult messages need their corresponding tool_use, skip them
    // assistant messages are ok to start from if they don't reference missing tool calls
    if (msg.role === "assistant") {
      // Check if previous messages have the required tool_use for any following tool_result
      const nextMsg = messages[start + 1];
      if (nextMsg && nextMsg.role === "user") {
        const nextAny = nextMsg as any;
        if (Array.isArray(nextAny.content)) {
          const hasToolResult = nextAny.content.some((b: any) => b.type === "tool_result");
          if (hasToolResult) {
            // This assistant message has tool_use that's needed by next message
            break;
          }
        }
      }
    }

    start++;
  }

  return start;
}

/**
 * Simple compression based on message count (legacy logic, maintains backward compatibility)
 */
export function compactMessagesByCount(
  messages: AgentMessage[],
  maxMessages: number,
  keepLast: number,
): CompactionResult | null {
  if (messages.length <= maxMessages) return null;

  const targetStart = messages.length - keepLast;
  const safeStart = findSafeCompactionPoint(messages, targetStart);

  // If we can't find a safe point, don't compact
  if (safeStart >= messages.length) {
    return null;
  }

  const kept = messages.slice(safeStart);

  // Don't compact if we'd keep almost everything anyway
  if (kept.length >= messages.length - 2) {
    return null;
  }

  return {
    kept,
    removedCount: messages.length - kept.length,
    reason: "count",
  };
}

/**
 * Token-based intelligent compression
 */
export function compactMessagesByTokens(
  messages: AgentMessage[],
  availableTokens: number,
  options?: {
    targetRatio?: number;
    minKeepMessages?: number;
  },
): CompactionResult | null {
  const result = compactMessagesTokenAware(messages, availableTokens, options);
  if (!result) return null;

  return {
    kept: result.kept,
    removedCount: result.removedCount,
    tokensRemoved: result.tokensRemoved,
    tokensKept: result.tokensKept,
    reason: "tokens",
  };
}

/** Synchronous compaction options (count/tokens modes) */
export type SyncCompactionOptions = {
  mode: "count" | "tokens";
  // count mode parameters
  maxMessages?: number | undefined;
  keepLast?: number | undefined;
  // tokens mode parameters
  contextWindowTokens?: number | undefined;
  systemPrompt?: string | undefined;
  reserveTokens?: number | undefined;
  targetRatio?: number | undefined;
  minKeepMessages?: number | undefined;
};

/** Summary compaction options (summary mode) */
export type SummaryCompactionOptions = {
  mode: "summary";
  // Required parameters
  model: Model<any>;
  apiKey: string;
  // tokens mode parameters (reused)
  contextWindowTokens?: number | undefined;
  systemPrompt?: string | undefined;
  reserveTokens?: number | undefined;
  targetRatio?: number | undefined;
  minKeepMessages?: number | undefined;
  // summary-specific parameters
  customInstructions?: string | undefined;
  previousSummary?: string | undefined;
  signal?: AbortSignal | undefined;
  maxChunkTokens?: number | undefined;
};

export type CompactionOptions = SyncCompactionOptions | SummaryCompactionOptions;

/**
 * Unified compaction entry point (synchronous version, for count/tokens modes)
 *
 * Selects compaction strategy based on mode
 */
export function compactMessages(
  messages: AgentMessage[],
  options: SyncCompactionOptions,
): CompactionResult | null {
  if (options.mode === "count") {
    return compactMessagesByCount(
      messages,
      options.maxMessages ?? 80,
      options.keepLast ?? 60,
    );
  }

  // Token mode
  const contextWindowTokens = options.contextWindowTokens ?? 200_000;
  const estimation = estimateTokenUsage({
    messages,
    systemPrompt: options.systemPrompt,
    contextWindowTokens,
    reserveTokens: options.reserveTokens,
  });

  // 检查是否需要压缩
  if (!shouldCompactTokens(estimation)) {
    return null;
  }

  return compactMessagesByTokens(messages, estimation.availableTokens, {
    targetRatio: options.targetRatio ?? COMPACTION_TARGET_RATIO,
    minKeepMessages: options.minKeepMessages ?? MIN_KEEP_MESSAGES,
  });
}

/**
 * Summary-based compaction (asynchronous version)
 *
 * Uses LLM to generate summary of historical messages
 */
export async function compactMessagesAsync(
  messages: AgentMessage[],
  options: SummaryCompactionOptions,
): Promise<CompactionResult | null> {
  const contextWindowTokens = options.contextWindowTokens ?? 200_000;
  const estimation = estimateTokenUsage({
    messages,
    systemPrompt: options.systemPrompt,
    contextWindowTokens,
    reserveTokens: options.reserveTokens,
  });

  // Check if compaction is needed
  if (!shouldCompactTokens(estimation)) {
    return null;
  }

  // Use chunked summary to handle very large history
  const result = await compactMessagesWithChunkedSummary({
    messages,
    model: options.model,
    apiKey: options.apiKey,
    availableTokens: estimation.availableTokens,
    targetRatio: options.targetRatio ?? COMPACTION_TARGET_RATIO,
    minKeepMessages: options.minKeepMessages ?? MIN_KEEP_MESSAGES,
    reserveTokens: options.reserveTokens ?? 2048,
    customInstructions: options.customInstructions,
    previousSummary: options.previousSummary,
    signal: options.signal,
    maxChunkTokens: options.maxChunkTokens,
  });

  if (!result) {
    return null;
  }

  return {
    kept: result.kept,
    removedCount: result.removedCount,
    tokensRemoved: result.tokensRemoved,
    tokensKept: result.tokensKept,
    summary: result.summary,
    reason: "summary",
  };
}
