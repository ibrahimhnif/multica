import { afterEach, describe, expect, it, vi } from "vitest";
import { AsyncAgent } from "./async-agent.js";

const subscribeCallbacks: Array<(event: any) => void> = [];
const internalRunState = { value: false };

const runMock = vi.fn(async () => ({ text: "", thinking: undefined, error: undefined }));
const runInternalMock = vi.fn(async () => ({ text: "", thinking: undefined, error: undefined }));
const flushSessionMock = vi.fn(async () => {});
const subscribeAllMock = vi.fn((fn: (event: any) => void) => {
  subscribeCallbacks.push(fn);
  return () => {};
});

vi.mock("./runner.js", () => ({
  Agent: class MockAgent {
    sessionId = "test-session";
    subscribeAll = subscribeAllMock;
    run = runMock;
    runInternal = runInternalMock;
    flushSession = flushSessionMock;
    get isInternalRun() {
      return internalRunState.value;
    }
    getMessages() {
      return [];
    }
    loadSessionMessages() {
      return [];
    }
    async ensureInitialized() {}
    getActiveTools() {
      return [];
    }
    reloadTools() {
      return [];
    }
    getSkillsWithStatus() {
      return [];
    }
    getEligibleSkills() {
      return [];
    }
    reloadSkills() {}
    setToolStatus() {
      return undefined;
    }
    getProfileId() {
      return undefined;
    }
    getAgentName() {
      return undefined;
    }
    setAgentName() {}
    getUserContent() {
      return undefined;
    }
    setUserContent() {}
    getAgentStyle() {
      return undefined;
    }
    setAgentStyle() {}
    reloadSystemPrompt() {}
    getProviderInfo() {
      return { provider: "test", model: "test-model" };
    }
    setProvider() {
      return { provider: "test", model: "test-model" };
    }
  },
}));

async function nextWithTimeout<T>(iter: AsyncIterator<T>, timeoutMs = 40): Promise<"timeout" | T> {
  return await Promise.race([
    iter.next().then((result) => (result.done ? "timeout" : result.value)),
    new Promise<"timeout">((resolve) => setTimeout(() => resolve("timeout"), timeoutMs)),
  ]);
}

describe("AsyncAgent internal flow", () => {
  afterEach(() => {
    subscribeCallbacks.length = 0;
    internalRunState.value = false;
    runMock.mockReset();
    runInternalMock.mockReset();
    flushSessionMock.mockReset();
    subscribeAllMock.mockClear();
    runMock.mockResolvedValue({ text: "", thinking: undefined, error: undefined });
    runInternalMock.mockResolvedValue({ text: "", thinking: undefined, error: undefined });
    flushSessionMock.mockResolvedValue(undefined);
  });

  it("filters internal events in direct subscribe stream", () => {
    const agent = new AsyncAgent();
    const events: Array<{ type: string }> = [];

    const unsubscribe = agent.subscribe((event) => {
      events.push(event as { type: string });
    });

    // subscribeAll is called twice:
    // 1) constructor for read() channel forwarding
    // 2) subscribe() for direct callback forwarding
    const subscribeCallback = subscribeCallbacks[1];
    expect(subscribeCallback).toBeDefined();

    internalRunState.value = true;
    subscribeCallback!({ type: "message_end" });
    expect(events).toHaveLength(0);

    internalRunState.value = false;
    subscribeCallback!({ type: "message_end" });
    expect(events).toHaveLength(1);

    unsubscribe();
    agent.close();
  });

  it("does not leak internal run errors to read() stream", async () => {
    runInternalMock.mockResolvedValueOnce({ text: "", thinking: undefined, error: "internal failed" });
    const agent = new AsyncAgent();
    const iter = agent.read()[Symbol.asyncIterator]();

    agent.writeInternal("test internal");
    await agent.waitForIdle();

    const value = await nextWithTimeout(iter);
    expect(value).toBe("timeout");

    agent.close();
  });

  it("does not leak internal run exceptions to read() stream", async () => {
    runInternalMock.mockRejectedValueOnce(new Error("internal exception"));
    const agent = new AsyncAgent();
    const iter = agent.read()[Symbol.asyncIterator]();

    agent.writeInternal("test internal");
    await agent.waitForIdle();

    const value = await nextWithTimeout(iter);
    expect(value).toBe("timeout");

    agent.close();
  });

  it("forwards assistant message stream (start/update/end) when writeInternal opts in", async () => {
    let resolveRunInternal: ((value: { text: string; thinking: undefined; error: undefined }) => void) | undefined;
    runInternalMock.mockImplementationOnce(
      () => new Promise((resolve) => {
        resolveRunInternal = resolve as typeof resolveRunInternal;
      }),
    );

    const agent = new AsyncAgent();
    const iter = agent.read()[Symbol.asyncIterator]();
    const streamCallback = subscribeCallbacks[0];
    expect(streamCallback).toBeDefined();

    agent.writeInternal("announce", { forwardAssistant: true });
    await Promise.resolve();

    internalRunState.value = true;
    streamCallback!({
      type: "message_start",
      message: { role: "assistant", content: [] },
    });
    streamCallback!({
      type: "message_update",
      message: { role: "assistant", content: [{ type: "text", text: "partial" }] },
    });
    streamCallback!({
      type: "message_end",
      message: { role: "user", content: [{ type: "text", text: "hidden internal prompt" }] },
    });
    streamCallback!({
      type: "message_end",
      message: { role: "assistant", content: [{ type: "text", text: "visible summary" }] },
    });

    const first = await nextWithTimeout(iter);
    expect(first).not.toBe("timeout");
    if (first !== "timeout") {
      expect((first as { type: string }).type).toBe("message_start");
      expect((first as { message: { role: string } }).message.role).toBe("assistant");
    }

    const second = await nextWithTimeout(iter);
    expect(second).not.toBe("timeout");
    if (second !== "timeout") {
      expect((second as { type: string }).type).toBe("message_update");
      expect((second as { message: { role: string } }).message.role).toBe("assistant");
    }

    const third = await nextWithTimeout(iter);
    expect(third).not.toBe("timeout");
    if (third !== "timeout") {
      expect((third as { type: string }).type).toBe("message_end");
      expect((third as { message: { role: string } }).message.role).toBe("assistant");
    }

    const fourth = await nextWithTimeout(iter);
    expect(fourth).toBe("timeout");

    resolveRunInternal!({ text: "", thinking: undefined, error: undefined });
    await agent.waitForIdle();
    internalRunState.value = false;
    agent.close();
  });
});
