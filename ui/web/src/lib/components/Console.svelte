<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { initPyodide, executeCode } from "$lib/pyodide";
    import { updateStores, pendingCommand } from "$lib/stores";

    let terminalEl: HTMLDivElement;
    let terminal: any;
    let fitAddon: any;
    let inputBuffer = "";
    let cursorPos = 0;
    let history: string[] = [];
    let historyIdx = -1;
    let ready = false;
    let multiline = false;
    let multilineBuffer = "";

    const PROMPT = ">>> ";
    const CONTINUATION = "... ";

    function currentPrompt() {
        return multiline ? CONTINUATION : PROMPT;
    }

    /** Redraw the current input line with cursor at cursorPos */
    function redrawLine() {
        const prompt = currentPrompt();
        // Move to start, clear line, write prompt + buffer, reposition cursor
        terminal.write("\r\x1b[K" + prompt + inputBuffer);
        // Move cursor back to correct position
        const backSteps = inputBuffer.length - cursorPos;
        if (backSteps > 0) {
            terminal.write(`\x1b[${backSteps}D`);
        }
    }

    onMount(async () => {
        const { Terminal } = await import("@xterm/xterm");
        const { FitAddon } = await import("@xterm/addon-fit");
        await import("@xterm/xterm/css/xterm.css");

        terminal = new Terminal({
            cursorBlink: true,
            fontSize: 13,
            fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
            theme: {
                background: "#1a1a2e",
                foreground: "#e0e0e0",
                cursor: "#4fc3f7",
            },
            convertEol: true,
        });
        fitAddon = new FitAddon();
        terminal.loadAddon(fitAddon);
        terminal.open(terminalEl);
        fitAddon.fit();

        terminal.writeln("livn interactive console");
        terminal.writeln("Initializing Pyodide…");

        // Init Pyodide
        try {
            await initPyodide((msg: string) => {
                terminal.writeln(msg);
            });
            ready = true;
            terminal.write(PROMPT);
        } catch (e) {
            terminal.writeln(`\r\nFailed to initialize: ${e}`);
            return;
        }

        // Use onData for full clipboard/IME/paste support
        terminal.onData((data: string) => {
            if (!ready) return;

            for (let i = 0; i < data.length; i++) {
                const ch = data[i];
                const code = ch.charCodeAt(0);

                if (ch === "\r" || ch === "\n") {
                    // Check if this is part of a multi-char paste with more content
                    const remaining = data.slice(i + 1);
                    // Skip \n after \r in \r\n sequences
                    if (ch === "\r" && remaining.startsWith("\n")) {
                        i++;
                    }
                    handleEnter(false);
                } else if (ch === "\x7f" || ch === "\b") {
                    // Backspace
                    if (cursorPos > 0) {
                        inputBuffer = inputBuffer.slice(0, cursorPos - 1) + inputBuffer.slice(cursorPos);
                        cursorPos--;
                        redrawLine();
                    }
                } else if (ch === "\x03") {
                    // Ctrl+C
                    terminal.write("^C\r\n");
                    inputBuffer = "";
                    cursorPos = 0;
                    multiline = false;
                    multilineBuffer = "";
                    terminal.write(PROMPT);
                } else if (ch === "\x01") {
                    // Ctrl+A — home
                    cursorPos = 0;
                    redrawLine();
                } else if (ch === "\x05") {
                    // Ctrl+E — end
                    cursorPos = inputBuffer.length;
                    redrawLine();
                } else if (ch === "\x0b") {
                    // Ctrl+K — kill to end of line
                    inputBuffer = inputBuffer.slice(0, cursorPos);
                    redrawLine();
                } else if (ch === "\x15") {
                    // Ctrl+U — kill to start of line
                    inputBuffer = inputBuffer.slice(cursorPos);
                    cursorPos = 0;
                    redrawLine();
                } else if (ch === "\x1b") {
                    // Escape sequence
                    if (data[i + 1] === "[") {
                        const seq = data[i + 2];
                        if (seq === "A") {
                            // Arrow up — history
                            i += 2;
                            if (history.length > 0) {
                                if (historyIdx === -1) historyIdx = history.length;
                                if (historyIdx > 0) {
                                    historyIdx--;
                                    inputBuffer = history[historyIdx];
                                    cursorPos = inputBuffer.length;
                                    redrawLine();
                                }
                            }
                        } else if (seq === "B") {
                            // Arrow down — history
                            i += 2;
                            if (historyIdx !== -1 && historyIdx < history.length - 1) {
                                historyIdx++;
                                inputBuffer = history[historyIdx];
                                cursorPos = inputBuffer.length;
                                redrawLine();
                            } else if (historyIdx === history.length - 1) {
                                historyIdx = -1;
                                inputBuffer = "";
                                cursorPos = 0;
                                redrawLine();
                            }
                        } else if (seq === "C") {
                            // Arrow right
                            i += 2;
                            if (cursorPos < inputBuffer.length) {
                                cursorPos++;
                                terminal.write("\x1b[C");
                            }
                        } else if (seq === "D") {
                            // Arrow left
                            i += 2;
                            if (cursorPos > 0) {
                                cursorPos--;
                                terminal.write("\x1b[D");
                            }
                        } else if (seq === "H") {
                            // Home
                            i += 2;
                            cursorPos = 0;
                            redrawLine();
                        } else if (seq === "F") {
                            // End
                            i += 2;
                            cursorPos = inputBuffer.length;
                            redrawLine();
                        } else if (seq === "3" && data[i + 3] === "~") {
                            // Delete key
                            i += 3;
                            if (cursorPos < inputBuffer.length) {
                                inputBuffer = inputBuffer.slice(0, cursorPos) + inputBuffer.slice(cursorPos + 1);
                                redrawLine();
                            }
                        } else {
                            // Unknown escape — skip
                            i += 2;
                        }
                    } else {
                        // Alt+key or bare escape — skip
                    }
                } else if (code >= 32) {
                    // Printable character (also handles pasted text char-by-char)
                    inputBuffer = inputBuffer.slice(0, cursorPos) + ch + inputBuffer.slice(cursorPos);
                    cursorPos++;
                    redrawLine();
                }
            }
        });

        // Handle resize
        resizeObserver = new ResizeObserver(() => {
            fitAddon?.fit();
        });
        resizeObserver.observe(terminalEl);
    });

    let resizeObserver: ResizeObserver | null = null;

    function handleEnter(shift: boolean) {
        if (shift) {
            multiline = true;
            multilineBuffer += inputBuffer + "\n";
            inputBuffer = "";
            cursorPos = 0;
            terminal.write("\r\n" + CONTINUATION);
            return;
        }

        terminal.write("\r\n");

        if (multiline) {
            multilineBuffer += inputBuffer;
            const code = multilineBuffer;
            multilineBuffer = "";
            multiline = false;
            inputBuffer = "";
            cursorPos = 0;
            historyIdx = -1;
            if (code.trim()) {
                history.push(code);
                runCode(code);
            } else {
                terminal.write(PROMPT);
            }
        } else {
            const code = inputBuffer;
            inputBuffer = "";
            cursorPos = 0;
            historyIdx = -1;
            if (code.trim()) {
                history.push(code);
                runCode(code);
            } else {
                terminal.write(PROMPT);
            }
        }
    }

    async function runCode(code: string) {
        ready = false;
        try {
            const { output, error, snapshot } = await executeCode(code);
            if (output) {
                terminal.write(output.replace(/\n/g, "\r\n"));
                if (!output.endsWith("\n")) terminal.write("\r\n");
            }
            if (error) {
                // Show last line of traceback (most useful)
                const lines = error.split("\n").filter((l) => l.trim());
                const lastLine = lines[lines.length - 1] || error;
                terminal.write(`\x1b[31m${lastLine}\x1b[0m\r\n`);
            }
            if (snapshot) {
                updateStores(snapshot);
            }
        } catch (e) {
            terminal.write(`\x1b[31mError: ${e}\x1b[0m\r\n`);
        }
        ready = true;
        terminal.write(PROMPT);
    }

    onDestroy(() => {
        unsubCmd();
        resizeObserver?.disconnect();
        terminal?.dispose();
    });

    // Subscribe to external command injection
    const unsubCmd = pendingCommand.subscribe((code) => {
        if (code && terminal && ready) {
            // Echo the command in the terminal
            const lines = code.split("\n");
            for (let i = 0; i < lines.length; i++) {
                const prefix = i === 0 ? PROMPT : CONTINUATION;
                terminal.write("\r\x1b[K" + prefix + lines[i] + "\r\n");
            }
            inputBuffer = "";
            cursorPos = 0;
            history.push(code);
            historyIdx = -1;
            pendingCommand.set(null);
            runCode(code);
        }
    });
</script>

<div class="console" bind:this={terminalEl}></div>

<style>
    .console {
        width: 100%;
        flex: 1;
        min-height: 0;
        overflow: hidden;
    }
    .console :global(.xterm) {
        height: 100%;
        padding: 4px;
    }
</style>
