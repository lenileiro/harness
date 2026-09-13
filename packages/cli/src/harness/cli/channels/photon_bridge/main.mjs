import { StringDecoder } from "node:string_decoder";
import { once } from "node:events";
import { pathToFileURL } from "node:url";

export async function runBridge({ app, imessage, text, input, output }) {
  const spaces = new Map();
  const emit = async (frame) => {
    const line = JSON.stringify(frame) + "\n";
    if (Buffer.byteLength(line) > 1024 * 1024) throw new Error("Frame too large");
    if (!output.write(line)) await once(output, "drain");
  };
  const inbound = async () => {
    for await (const [space, message] of app.messages) {
      if (message.direction !== "inbound" || message.content?.type !== "text") continue;
      if (!message.id || !message.sender?.id || !space.id) continue;
      spaces.set(space.id, space);
      if (spaces.size > 1000) spaces.delete(spaces.keys().next().value);
      await emit({ event: "message", id: message.id, sender: message.sender.id,
        space: space.id, group: space.type !== "dm", text: message.content.text });
    }
  };
  const outbound = async () => {
    let buffer = "";
    const decoder = new StringDecoder("utf8");
    for await (const chunk of input) {
      buffer += decoder.write(chunk);
      if (Buffer.byteLength(buffer) > 1024 * 1024) throw new Error("Command too large");
      let newline;
      while ((newline = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, newline); buffer = buffer.slice(newline + 1);
        let frame;
        try {
          frame = JSON.parse(line);
          if (frame.method !== "send" || typeof frame.id !== "string" || typeof frame.space !== "string" || typeof frame.text !== "string" || frame.text.length > 8000) throw new Error("Invalid command");
          const space = spaces.get(frame.space) || await imessage(app).space.get(frame.space);
          if (!space || space.id !== frame.space) throw new Error("Wrong space");
          const result = await space.send(text(frame.text));
          if (!result?.id) throw new Error("No receipt");
          await emit({ id: frame.id, result: { message_id: result.id } });
        } catch {
          await emit({ id: typeof frame?.id === "string" ? frame.id : null, error: "Photon send failed; outcome may be uncertain" });
        }
      }
    }
  };
  await Promise.race([inbound(), outbound()]);
}

async function main() {
  // SDK diagnostics never share the framed stdout or expose credentials in stderr.
  console.log = console.info = console.warn = console.error = () => {};
  const { Spectrum, text } = await import("spectrum-ts");
  const { imessage } = await import("spectrum-ts/providers/imessage");
  const app = await Spectrum({ projectId: process.env.PHOTON_PROJECT_ID,
    projectSecret: process.env.PHOTON_PROJECT_SECRET, providers: [imessage.config()],
    options: { flattenGroups: true } });
  process.stdout.write(JSON.stringify({ event: "ready", project: process.env.PHOTON_PROJECT_ID }) + "\n");
  await runBridge({ app, imessage, text, input: process.stdin, output: process.stdout });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().then(() => process.exit(0)).catch(() => {
    process.stderr.write("Photon bridge failed; verify Node/SDK installation and project credentials.\n");
    process.exit(1);
  });
}
