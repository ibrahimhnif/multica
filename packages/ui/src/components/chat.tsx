"use client";

import { useRef, useState, useCallback, useMemo } from "react";
import { Button } from "@multica/ui/components/ui/button";
import { Textarea } from "@multica/ui/components/ui/textarea";
import { ChatInput } from "@multica/ui/components/chat-input";
import { MemoizedMarkdown } from "@multica/ui/components/markdown";
import { StreamingMarkdown } from "@multica/ui/components/markdown/StreamingMarkdown";
import { toast } from "@multica/ui/components/ui/sonner";
import {
  useHubStore,
  useMessagesStore,
  useGatewayStore,
  useHubInit,
  useDeviceId,
  parseConnectionCode,
  saveConnection,
  clearConnection,
} from "@multica/store";
import { useScrollFade } from "@multica/ui/hooks/use-scroll-fade";
import { useAutoScroll } from "@multica/ui/hooks/use-auto-scroll";
import { cn } from "@multica/ui/lib/utils";

interface ChatProps {
  showHeader?: boolean;
}

export function Chat({ showHeader = true }: ChatProps) {
  useHubInit()
  const deviceId = useDeviceId()
  const activeAgentId = useHubStore((s) => s.activeAgentId)
  const gwState = useGatewayStore((s) => s.connectionState)
  const hubId = useGatewayStore((s) => s.hubId)

  const messages = useMessagesStore((s) => s.messages)
  const streamingIds = useMessagesStore((s) => s.streamingIds)
  const filtered = useMemo(() => messages.filter(m => m.agentId === activeAgentId), [messages, activeAgentId])

  const isConnected = gwState === "registered" && !!hubId && !!activeAgentId
  const [codeInput, setCodeInput] = useState("")

  const handleConnect = useCallback(() => {
    const trimmed = codeInput.trim()
    if (!trimmed) return
    try {
      const info = parseConnectionCode(trimmed)
      saveConnection(info)
      useGatewayStore.getState().connectWithCode(info, deviceId)
      setCodeInput("")
    } catch (e) {
      toast.error((e as Error).message)
    }
  }, [codeInput, deviceId])

  const handleDisconnect = useCallback(() => {
    useGatewayStore.getState().disconnect()
    useHubStore.setState({ status: "idle", hub: null, agents: [], activeAgentId: null })
    clearConnection()
  }, [])

  const handleSend = useCallback((text: string) => {
    const { hubId } = useGatewayStore.getState()
    const agentId = useHubStore.getState().activeAgentId
    if (!hubId || !agentId) return
    useMessagesStore.getState().addUserMessage(text, agentId)
    useGatewayStore.getState().send(hubId, "message", { agentId, content: text })
  }, [])

  const mainRef = useRef<HTMLElement>(null)
  const fadeStyle = useScrollFade(mainRef)
  useAutoScroll(mainRef)

  return (
    <div className="h-full flex flex-col overflow-hidden w-full">
      {/* Header */}
      {showHeader && (
        <header>
          <div className="flex items-center justify-between px-4 py-2 max-w-4xl mx-auto">
            <div className="flex items-center gap-2.5">
              <img src="/icon.png" alt="Multica" className="size-6 rounded-md" />
              <span className="text-sm tracking-wide font-[family-name:var(--font-brand)]">
                Multica
              </span>
            </div>
            <div className="flex items-center gap-1">
              {isConnected && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleDisconnect}
                  className="text-xs text-muted-foreground"
                >
                  Disconnect
                </Button>
              )}
            </div>
          </div>
        </header>
      )}

      {/* Main */}
      <main ref={mainRef} className="flex-1 overflow-y-auto min-h-0" style={fadeStyle}>
        {!isConnected ? (
          <div className="flex flex-col items-center justify-center h-full gap-4 px-4">
            <div className="text-center space-y-1">
              <p className="text-sm text-muted-foreground">Paste a connection code to start</p>
              {(gwState === "connecting" || gwState === "connected") && (
                <p className="text-xs text-muted-foreground/60 animate-pulse">Connecting...</p>
              )}
            </div>
            <div className="w-full max-w-sm space-y-3">
              <Textarea
                value={codeInput}
                onChange={(e) => setCodeInput(e.target.value)}
                placeholder="Paste connection code here..."
                className="text-xs font-mono min-h-[100px] resize-none"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault()
                    handleConnect()
                  }
                }}
              />
              <Button
                size="sm"
                onClick={handleConnect}
                disabled={!codeInput.trim() || gwState === "connecting"}
                className="w-full text-xs"
              >
                Connect
              </Button>
            </div>
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
            Send a message to start the conversation
          </div>
        ) : (
          <div className="px-4 py-6 space-y-6 max-w-4xl mx-auto">
            {filtered.map((msg) => {
              const isStreaming = streamingIds.has(msg.id)
              return (
                <div
                  key={msg.id}
                  className={cn(
                    "flex",
                    msg.role === "user" ? "justify-end" : "justify-start"
                  )}
                >
                  <div
                    className={cn(
                      msg.role === "user" ? "bg-muted rounded-md max-w-[60%] p-1 px-2.5" : "w-full p-1 px-2.5"
                    )}
                  >
                    {isStreaming ? (
                      <StreamingMarkdown content={msg.content} isStreaming={true} mode="minimal" />
                    ) : (
                      <MemoizedMarkdown mode="minimal" id={msg.id}>
                        {msg.content}
                      </MemoizedMarkdown>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="w-full p-2 pt-1 max-w-4xl mx-auto">
        <ChatInput
          onSubmit={handleSend}
          disabled={!isConnected}
          placeholder={!isConnected ? "Connect first..." : "Type a message..."}
        />
      </footer>
    </div>
  );
}
