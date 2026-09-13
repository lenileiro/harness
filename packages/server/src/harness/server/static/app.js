"use strict";
const $ = (id) => document.getElementById(id);
const state = { token: "", session: null, run: null, stream: null, cursor: 0, attachments: [], view: "conversation", streamText: "", epoch: 0, questions: [], questionDrafts: new Map(), uploadEpoch: 0, uploading: false };
const terminal = new Set(["completed", "paused", "failed", "cancelled", "interrupted"]);
function element(tag, text, className) { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (className) node.className = className; return node; }
function notice(text, error = false) { $("notice").textContent = text; $("notice").className = error ? "notice error" : "notice"; $("notice").hidden = !text; }
async function api(path, options = {}) {
  const token = state.token;
  const response = await fetch(path, { ...options, headers: { Authorization: `Bearer ${state.token}`, ...(options.body ? { "Content-Type": "application/json" } : {}), ...options.headers } });
  if (state.token !== token) throw new DOMException("Identity changed", "AbortError");
  if (!response.ok) { let detail; try { detail = (await response.json()).detail; } catch { detail = `Request failed (${response.status})`; } throw new Error(detail || `Request failed (${response.status})`); }
  return response;
}
const get = async (path) => { const token=state.token; const value=await (await api(path)).json(); if(token!==state.token) throw new DOMException("Identity changed","AbortError"); return value; };
const post = async (path, data = {}) => { const token=state.token; const value=await (await api(path,{method:"POST",body:JSON.stringify(data)})).json(); if(token!==state.token) throw new DOMException("Identity changed","AbortError"); return value; };
function action(label, callback, className = "secondary") { const button = element("button", label, className); button.type = "button"; button.onclick = async () => { button.disabled = true; try { await callback(); } catch (error) { notice(error.message, true); } finally { button.disabled = false; } }; return button; }
function stopStream() { if (state.stream) state.stream.abort(); state.stream = null; state.epoch++; state.uploadEpoch++; state.uploading=false; $("files").disabled=false; }
function updateRun(run) {
  state.run = run; const value = run ? run.state : "Ready";
  $("run-state").textContent = value; $("run-state").className = `state ${value}`;
  $("cancel-run").hidden = !run || terminal.has(value);
  $("resume-run").hidden = !run || !terminal.has(value) || value === "cancelled" || (run.kind === "tool" && value !== "paused");
  $("review-input").hidden = true;
  $("send").disabled = state.uploading || (!!run && !terminal.has(value));
  $("export").hidden = !state.session || state.view !== "conversation";
}
function activate(view, title) { state.view = view; $("run-state").hidden = view !== "conversation"; $("export").hidden = view !== "conversation" || !state.session; $("conversation-view").hidden = view !== "conversation"; $("list-view").hidden = view === "conversation"; $("view-title").textContent = title; document.querySelectorAll(".nav-item").forEach(button => button.classList.remove("active")); const selected = {conversation:"show-conversations", runs:"show-runs", inbox:"show-inbox", questions:"show-questions", batches:"show-batches", jobs:"show-jobs", schedules:"show-schedules", tools:"show-tools", settings:"show-settings"}[view]; $(selected).classList.add("active"); $("sidebar").classList.remove("mobile-open"); }
function attachmentNode(attachment) {
  const url = attachment.data ? `data:${attachment.mime_type};base64,${attachment.data}` : attachment.url;
  // Remote content opens only after an explicit click; viewing a transcript never contacts third parties.
  if (attachment.data && attachment.kind === "image") { const image = element("img"); image.src = url; image.alt = attachment.name || "Attached image"; return image; }
  if (attachment.data && attachment.kind === "audio") { const audio = element("audio"); audio.controls = true; audio.src = url; return audio; }
  const link = element("a", attachment.name || `Attached ${attachment.kind}`); link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer"; if (attachment.data) link.download = attachment.name || "attachment"; return link;
}
function messageNode(message) { const node = element("article", undefined, `message ${message.role}`); node.append(element("div", message.name || message.role, "message-role")); node.append(element("div", message.content || "", "message-body")); for (const call of message.tool_calls || []) node.append(element("pre", `${call.name}\n${JSON.stringify(call.arguments, null, 2)}`)); for (const attachment of message.attachments || []) node.append(attachmentNode(attachment)); return node; }
async function loadMessages() { if (!state.session) return; const session = state.session; const epoch=state.epoch; const data = await get(`/v1/sessions/${encodeURIComponent(session)}/messages`); if (session !== state.session || epoch !== state.epoch) return; $("messages").replaceChildren(...data.messages.map(messageNode)); $("messages").scrollTop = $("messages").scrollHeight; }
async function refreshSidebar() {
  const query = $("search").value.trim(); const data = await get(`/v1/sessions${query ? `?q=${encodeURIComponent(query)}` : ""}`);
  $("session-count").textContent = data.sessions.length;
  $("sessions").replaceChildren(...data.sessions.map(session => { const id = session.id || session.session_id; const button = element("button", undefined, `session ${id === state.session ? "selected" : ""}`); button.append(element("strong", session.excerpt || id.slice(-12))); button.append(element("small", session.status || "Search match")); button.onclick = () => openSession(id).catch(error => notice(error.message, true)); return button; }));
  const [inbox, questions] = await Promise.all([get("/v1/approvals"), get("/v1/questions")]); $("approval-count").textContent = inbox.approvals.length; state.questions = questions.questions; $("question-count").textContent = state.questions.length;
}
async function openSession(id) {
  stopStream(); const epoch=state.epoch; notice("");
  if (state.session!==id) {state.attachments=[];renderAttachments();$("prompt").value="";}
  state.session=id;state.cursor=0;updateRun(null);$("events").replaceChildren();$("activity-count").textContent="0";
  activate("conversation", `Conversation ${id.slice(-8)}`);
  const current=()=>state.epoch===epoch&&state.session===id;
  const runs=await get(`/v1/runs?session_id=${encodeURIComponent(id)}`);if(!current())return;
  updateRun(runs.runs[0]||null);await loadMessages();if(!current())return;
  await refreshSidebar();if(current()&&state.run)connectStream(state.run.id);
}

function newConversation() { stopStream(); state.session = null; state.cursor = 0; state.attachments = []; renderAttachments(); updateRun(null); notice(""); activate("conversation", "New conversation"); $("messages").replaceChildren(element("div", "What would you like to work on?", "empty")); $("events").replaceChildren(); $("activity-count").textContent = "0"; $("prompt").value = ""; $("prompt").focus(); }
function eventReceived(event) {
  if (event.type === "run_status") updateRun({ ...state.run, state: event.state, error: event.error });
  if (event.type === "text_delta") { state.streamText += event.text; let preview = $("stream-preview"); if (!preview) { preview = messageNode({role:"assistant", content:""}); preview.id = "stream-preview"; $("messages").append(preview); } preview.querySelector(".message-body").textContent = state.streamText; $("messages").scrollTop = $("messages").scrollHeight; }
  if (event.type === "error" || event.error) notice(event.error, true);
  if (event.type !== "text_delta" && event.type !== "model_request") { $("events").append(element("div", JSON.stringify(event, null, 2), "event")); $("activity-count").textContent = $("events").childElementCount; }
}
async function connectStream(runId) {
  if (state.stream) state.stream.abort(); const controller = new AbortController(); state.stream = controller; const epoch = state.epoch; const current=()=>epoch===state.epoch&&state.stream===controller&&!controller.signal.aborted&&state.run?.id===runId; state.streamText = ""; $("reconnect-run").hidden = true;
  try {
    const response = await api(`/v1/runs/${encodeURIComponent(runId)}/events?after=${state.cursor}`, {signal:controller.signal}); const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = "";
    while (true) { const {value, done} = await reader.read(); if (done) break; if (!current()) { await reader.cancel(); return; } buffer += decoder.decode(value, {stream:true}); let boundary; while ((boundary = buffer.indexOf("\n\n")) !== -1) { const block = buffer.slice(0,boundary); buffer = buffer.slice(boundary+2); let payload = ""; for (const line of block.split("\n")) { if (line.startsWith("id: ")) state.cursor = Number(line.slice(4)); if (line.startsWith("data: ")) payload += line.slice(6); } if (payload) eventReceived(JSON.parse(payload)); } }
    if (!current()) return; const run = await get(`/v1/runs/${encodeURIComponent(runId)}`); if(!current())return; updateRun(run); await loadMessages(); if(!current())return; await refreshSidebar(); if(!current())return; if (!terminal.has(run.state)) { notice("The event stream disconnected. Reconnect to continue following this run."); $("reconnect-run").hidden = false; } else if (run.state === "paused") await describePausedRun(run);
  } catch (error) { if (error.name !== "AbortError" && current()) { notice(error.message, true); $("reconnect-run").hidden = false; } }
}
function card(title, description, status) { const item = element("article", undefined, "row-card"); const top = element("div", undefined, "row-top"); top.append(element("h2", title)); if (status) top.append(element("span", status, `state ${status}`)); item.append(top); if (description) item.append(element("p", description)); return item; }
async function showRuns() { activate("runs", "Runs"); $("list-toolbar").replaceChildren(); const data = await get("/v1/runs"); $("list-content").replaceChildren(...data.runs.map(run => { const item = card(run.delegation_handle ? `Child ${run.delegation_handle.slice(-8)}` : run.id.slice(-12), new Date(run.created_at).toLocaleString(), run.state); item.append(action("Open conversation", () => openSession(run.session_id))); if (run.error) item.append(element("p", run.error)); return item; })); if (!data.runs.length) $("list-content").append(element("p", "Your runs will appear here.", "empty-list")); }
async function showInbox() { activate("inbox", "Approvals"); $("list-toolbar").replaceChildren(element("p", "Review each action before it changes the workspace or an external service.", "muted")); const data = await get("/v1/approvals"); $("list-content").replaceChildren(...data.approvals.map(approval => { const item = card(approval.tool_name, `Conversation ${approval.session_id.slice(-8)}`, "paused"); item.append(element("pre", JSON.stringify(approval.arguments, null, 2))); const actions = element("div", undefined, "row-actions"); const resolve = async granted => { await post(`/v1/approvals/${encodeURIComponent(approval.id)}/resolve`, {granted}); notice(granted ? "Approved. Open the conversation and resume its run to execute the action." : "Action denied."); await showInbox(); await refreshSidebar(); }; actions.append(action("Approve", () => resolve(true), "primary"), action("Deny", () => resolve(false), "danger"), action("Open conversation", () => openSession(approval.session_id), "quiet")); item.append(actions); return item; })); if (!data.approvals.length) $("list-content").append(element("p", "No actions are waiting for your approval.", "empty-list")); }
async function showBatches() { activate("batches", "Batches"); $("list-toolbar").replaceChildren(action("＋ Create batch", () => $("batch-dialog").showModal(), "primary")); const data = await get("/v1/batches"); $("list-content").replaceChildren(...data.batches.map(batch => { const done = batch.runs.filter(run => terminal.has(run.state)).length; const item = card(`Batch ${batch.id.slice(-8)}`, `${done} of ${batch.runs.length} finished`, batch.complete ? "completed" : "running"); for (const run of batch.runs) item.append(action(`${run.state} · ${run.id.slice(-8)}`, () => openSession(run.session_id), "quiet")); if (!batch.complete) item.append(action("Cancel remaining runs", async () => { await post(`/v1/batches/${encodeURIComponent(batch.id)}/cancel`); await showBatches(); }, "danger")); return item; })); if (!data.batches.length) $("list-content").append(element("p", "Queue independent tasks and track them together.", "empty-list")); }
function renderAttachments() { $("attachments").replaceChildren(...state.attachments.map((attachment, index) => { const chip = element("span", attachment.name, "attachment-chip"); chip.append(action("×", () => { state.attachments.splice(index,1); renderAttachments(); }, "quiet")); return chip; })); $("prompt").required = !state.attachments.length; }
$("files").onchange = async () => {
  const files=[...$("files").files];const epoch=state.epoch;const upload=++state.uploadEpoch;const token=state.token;
  const current=()=>epoch===state.epoch&&upload===state.uploadEpoch&&token===state.token;
  state.uploading=true;$("files").disabled=true;$("send").disabled=true;
  try {
    if(state.attachments.length+files.length>16||files.some(file=>file.size>8*1024*1024))throw new Error("Attach at most 16 files, each no larger than 8 MiB.");
    const additions=[];
    for(const file of files){
      const bytes=new Uint8Array(await file.arrayBuffer());if(!current())return;
      let binary="";for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192));
      const mime=file.type||"application/octet-stream";
      additions.push({version:1,kind:mime.startsWith("image/")?"image":mime.startsWith("audio/")?"audio":"file",mime_type:mime,name:file.name,data:btoa(binary)});
    }
    if(current()){state.attachments.push(...additions);renderAttachments();}
  } catch(error){if(current())notice(error.message,true);}
  finally{if(current()){state.uploading=false;$("files").disabled=false;$("files").value="";$("send").disabled=!!state.run&&!terminal.has(state.run.state);}}
};
$("compose").onsubmit = async event => {
  event.preventDefault();if(state.uploading||(state.run&&!terminal.has(state.run.state)))return;
  const epoch=state.epoch;const token=state.token;const prompt=$("prompt").value;const attachments=[...state.attachments];
  $("send").disabled=true;
  try {
    const run=await post("/v1/runs",{prompt,session_id:state.session,attachments});
    if(epoch!==state.epoch){if(token===state.token){await refreshSidebar();notice("Run queued. You can find it in Runs.");}return;}
    stopStream();state.session=run.session_id;state.cursor=0;updateRun(run);notice("");
    if(!$("messages").querySelector(".message"))$("messages").replaceChildren();
    $("messages").append(messageNode({role:"user",content:prompt,attachments}));$("prompt").value="";state.attachments=[];renderAttachments();
    $("events").replaceChildren();$("activity-count").textContent="0";activate("conversation",`Conversation ${run.session_id.slice(-8)}`);
    connectStream(run.id);await refreshSidebar();
  }catch(error){if(epoch===state.epoch&&token===state.token){notice(error.message,true);$("send").disabled=state.uploading||(!!state.run&&!terminal.has(state.run.state));}}
};

$("prompt").onkeydown = event => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!$("send").disabled) $("compose").requestSubmit(); } };
$("batch-form").onsubmit = async event => { event.preventDefault(); try { const runs=$("batch-prompts").value.split("\n").map(prompt=>prompt.trim()).filter(Boolean).map(prompt=>({prompt})); await post("/v1/batches",{runs}); $("batch-dialog").close(); $("batch-prompts").value=""; await showBatches(); await refreshSidebar(); } catch(error) { notice(error.message,true); $("batch-dialog").close(); } };
window.harnessConnect = async token => { stopStream(); newConversation(); state.questions=[]; state.questionDrafts.clear(); state.token=token; try { const health=await get("/v1/health"); $("identity").textContent=health.user_id; $("token").value=""; $("login").hidden=true; $("workspace").hidden=false; $("login-error").textContent=""; await refreshSidebar(); } catch(error) { state.token=""; $("login-error").textContent=error.message; $("login").hidden=false; $("workspace").hidden=true; } };
$("login-form").onsubmit=event=>{event.preventDefault(); window.harnessConnect($("token").value.trim());};
$("disconnect").onclick=()=>{stopStream(); state.token=""; state.session=null; state.run=null; state.attachments=[]; state.questions=[]; state.questionDrafts.clear(); $("messages").replaceChildren(); $("sessions").replaceChildren(); $("events").replaceChildren(); $("list-content").replaceChildren(); $("workspace").hidden=true; $("login").hidden=false; $("token").focus();};
$("new-chat").onclick=newConversation;
$("show-conversations").onclick=()=>state.session?openSession(state.session).catch(error=>notice(error.message,true)):newConversation();
$("show-runs").onclick=()=>showRuns().catch(error=>notice(error.message,true));
$("show-inbox").onclick=()=>showInbox().catch(error=>notice(error.message,true));
$("show-questions").onclick=()=>showQuestions().catch(error=>notice(error.message,true));
$("show-batches").onclick=()=>showBatches().catch(error=>notice(error.message,true));
$("refresh").onclick=async()=>{try {await refreshSidebar(); if(state.view==="runs") await showRuns(); else if(state.view==="inbox") await showInbox(); else if(state.view==="questions") await showQuestions(); else if(state.view==="batches") await showBatches(); else if(state.view==="jobs") await showJobs(); else if(state.view==="schedules") await showSchedules(); else if(state.view==="tools") await showTools(); else if(state.view==="settings") await showSettings(); else if(state.session) await openSession(state.session);} catch(error){notice(error.message,true);}};
$("cancel-run").onclick=async()=>{
  const target=state.run;const epoch=state.epoch;if(!target)return;
  try {const run=await post(`/v1/runs/${target.id}/cancel`);if(epoch===state.epoch&&state.run?.id===target.id)updateRun(run);}
  catch(error){if(epoch===state.epoch)notice(error.message,true);}
};
$("resume-run").onclick=async()=>{
  const target=state.run;const epoch=state.epoch;if(!target)return;
  try {
    const pending=(await get("/v1/questions")).questions.find(question=>question.session_id===target.session_id&&question.status==="pending");
    if(epoch!==state.epoch||state.run?.id!==target.id)return;
    if(pending){await showQuestions();notice("Answer the pending questions or skip them before continuing.");return;}
    const run=await post(`/v1/runs/${target.id}/resume`);if(epoch!==state.epoch||state.run?.id!==target.id)return;
    stopStream();state.cursor=0;updateRun(run);notice("");$("events").replaceChildren();$("activity-count").textContent="0";connectStream(run.id);
  }catch(error){if(epoch===state.epoch)notice(error.message,true);}
};

$("reconnect-run").onclick=()=>state.run&&connectStream(state.run.id);
$("export").onclick=async()=>{try {const response=await api(`/v1/sessions/${state.session}/export`); const url=URL.createObjectURL(await response.blob()); const link=element("a"); link.href=url; link.download=`${state.session}.jsonl`; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);} catch(error){notice(error.message,true);}};
$("close-batch").onclick=()=>$("batch-dialog").close();
let searchTimer; $("search").oninput=()=>{clearTimeout(searchTimer); searchTimer=setTimeout(()=>refreshSidebar().catch(error=>notice(error.message,true)),250);};
$("menu").onclick=()=>$("sidebar").classList.toggle("mobile-open");

async function showJobs() {
  activate("jobs", "Delegated jobs");
  $("list-toolbar").replaceChildren(state.session ? action("＋ Delegate from this conversation", () => $("job-dialog").showModal(), "primary") : element("p", "Open a conversation first to delegate work from it.", "muted"));
  const data = await get("/v1/delegations");
  $("list-content").replaceChildren(...data.jobs.map(job => {
    const item=card(`Child ${job.id.slice(-8)}`, `${job.attempts} attempt(s) · ${job.budget.steps} model calls reserved across this parent’s children`, job.state);
    if(job.summary) item.append(element("p",job.summary));
    if(job.run.error) item.append(element("p",job.run.error));
    const actions=element("div",undefined,"row-actions");
    actions.append(action("Open conversation",()=>openSession(job.session_id)));
    if(terminal.has(job.state)) actions.append(action("Resume",async()=>{await post(`/v1/delegations/${job.id}/resume`); await openSession(job.session_id);},"quiet"));
    else actions.append(action("Cancel",async()=>{await post(`/v1/delegations/${job.id}/cancel`); await showJobs();},"danger"));
    item.append(actions);
    for(const path of job.artifacts) item.append(action(`Download ${path}`,async()=>{const response=await api(`/v1/delegations/${job.id}/artifacts?path=${encodeURIComponent(path)}`); const url=URL.createObjectURL(await response.blob()); const link=element("a"); link.href=url; link.download=path.split("/").pop(); link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);},"quiet"));
    return item;
  }));
  if(!data.jobs.length) $("list-content").append(element("p","Independent child agents and their results will appear here.","empty-list"));
}
$("show-jobs").onclick=()=>showJobs().catch(error=>notice(error.message,true));
$("close-job").onclick=()=>$("job-dialog").close();
$("job-form").onsubmit=async event=>{event.preventDefault();try{await post("/v1/delegations",{parent_session_id:state.session,prompt:$("job-prompt").value,inputs:$("job-inputs").value.split("\n").map(path=>path.trim()).filter(Boolean)});$("job-dialog").close();$("job-prompt").value="";$("job-inputs").value="";await showJobs();await refreshSidebar();}catch(error){$("job-dialog").close();notice(error.message,true);}};


async function showSettings() {
  activate("settings", "Preferences");
  $("list-toolbar").replaceChildren(element("p", "Choose defaults for new conversations, batches and schedules. Existing conversations keep their selected model.", "muted"));
  const [config, preferences] = await Promise.all([get("/v1/configuration"), get("/v1/preferences")]);
  if(state.view!=="settings") return;
  const form=element("form",undefined,"settings-form row-card");
  const label=(text,id)=>{const node=element("label",text);node.htmlFor=id;return node;};
  const provider=element("select");provider.id="preference-provider";
  const fallback=element("option","Server default");fallback.value="";provider.append(fallback);
  for(const option of config.providers){const node=element("option",option.id);node.value=option.id;provider.append(node);}
  provider.value=preferences.provider||"";
  const model=element("input");model.id="preference-model";model.value=preferences.model||"";model.placeholder=config.default_model||"Provider model identifier";model.maxLength=256;
  provider.onchange=()=>{model.value="";model.placeholder=config.providers.find(item=>item.id===provider.value)?.default_model||"Provider model identifier";};
  const timezone=element("input");timezone.id="preference-timezone";timezone.value=preferences.timezone||"UTC";timezone.required=true;
  const button=element("button","Save preferences","primary");button.type="submit";
  form.append(label("Provider","preference-provider"),provider,label("Model (blank uses provider default)","preference-model"),model,label("Timezone","preference-timezone"),timezone,element("p","Only providers enabled by the server are available here. Credentials and tool permissions are managed by its operator.","hint"),button);
  form.onsubmit=async event=>{event.preventDefault();button.disabled=true;try{await post("/v1/preferences",{provider:provider.value||null,model:model.value.trim()||null,timezone:timezone.value.trim()});notice("Preferences saved for this identity.");}catch(error){notice(error.message,true);}finally{button.disabled=false;}};
  $("list-content").replaceChildren(form);
}
async function showTools() {
  activate("tools","Tools");
  $("list-toolbar").replaceChildren(element("p","Inspect the available tool schemas and server exposure policy. Exposure still passes through Harness approvals and evidence checks.","muted"));
  const data=await get("/v1/tools");if(state.view!=="tools") return;
  $("list-content").replaceChildren(...data.tools.map(tool=>{const item=card(tool.name,tool.description,tool.exposed?"Exposed":"Not exposed");item.append(element("p",`Effect: ${{read_only:"Reads information",session_ephemeral:"Temporary session changes",workspace_durable:"Changes workspace files",task_durable:"Changes saved work",external_side_effect:"Changes an external service",agent_orchestration:"Starts child agents"}[tool.effect_scope]||tool.effect_scope} · Approval: ${tool.effective_approval}`));const details=element("details");details.append(element("summary","Parameter schema"),element("pre",tool.parameters_schema?JSON.stringify(tool.parameters_schema,null,2):"Available when its runtime session starts."));item.append(details);return item;}));
  if(!data.tools.length) $("list-content").append(element("p","The server has not published any tools.","empty-list"));
}
function intervalLabel(seconds) { for(const [unit,size] of [["day",86400],["hour",3600],["minute",60]]) {if(seconds%size===0) {const count=seconds/size;return `Every ${count} ${unit}${count===1?"":"s"}`;}} return `Every ${seconds} seconds`; }
async function showSchedules() {
  activate("schedules","Schedules");
  $("list-toolbar").replaceChildren(action("＋ Schedule a prompt",async()=>{const preferences=await get("/v1/preferences");$("schedule-timezone").value=preferences.timezone;$("schedule-dialog").showModal();},"primary"));
  const data=await get("/v1/schedules");if(state.view!=="schedules") return;
  $("list-content").replaceChildren(...data.schedules.map(schedule=>{
    const latest=schedule.runs[0]?.run;
    const timing=schedule.schedule.kind==="every"?intervalLabel(Number(schedule.schedule.value)):schedule.schedule.value;
    const item=card(schedule.title,`${timing} · ${schedule.schedule.timezone}`,schedule.state);
    item.append(element("p",schedule.request.prompt));
    item.append(element("p",`${schedule.request.provider||"Server provider"} / ${schedule.request.model||"Default model"} · ${schedule.request.max_steps} steps maximum`));
    if(schedule.next_run_at&&schedule.state==="active") item.append(element("p",`Next: ${new Date(schedule.next_run_at).toLocaleString([], {timeZone:schedule.schedule.timezone})}`));
    if(schedule.error) item.append(element("p",schedule.error));
    if(latest) item.append(element("p",`Latest run: ${latest.state}`));
    const actions=element("div",undefined,"row-actions");
    if(schedule.state==="active") actions.append(action("Pause",async()=>{await post(`/v1/schedules/${schedule.id}/pause`);await showSchedules();},"quiet"));
    if(["paused","error"].includes(schedule.state)) actions.append(action("Resume schedule",async()=>{await post(`/v1/schedules/${schedule.id}/resume`);await showSchedules();}));
    if(!["cancelled","completed"].includes(schedule.state)||(latest&&!terminal.has(latest.state))) actions.append(action("Cancel schedule",async()=>{await post(`/v1/schedules/${schedule.id}/cancel`);await showSchedules();},"danger"));
    if(latest) actions.append(action("Open latest run",()=>openSession(latest.session_id)));
    item.append(actions);return item;
  }));
  if(!data.schedules.length) $("list-content").append(element("p","Schedule a recurring review or a one-time task. Runs and approvals stay here.","empty-list"));
}
$("show-settings").onclick=()=>showSettings().catch(error=>notice(error.message,true));
$("show-tools").onclick=()=>showTools().catch(error=>notice(error.message,true));
$("show-schedules").onclick=()=>showSchedules().catch(error=>notice(error.message,true));
$("close-schedule").onclick=()=>$("schedule-dialog").close();
$("schedule-kind").onchange=()=>{const kind=$("schedule-kind").value;$("schedule-value").type=kind==="at"?"datetime-local":"text";$("schedule-value").value=kind==="every"?"1h":kind==="cron"?"0 9 * * 1-5":"";$("schedule-value-label").textContent=kind==="every"?"Interval":kind==="cron"?"Cron expression":"Local date and time";};
$("schedule-form").onsubmit=async event=>{event.preventDefault();const button=event.submitter;button.disabled=true;try{await post("/v1/schedules",{title:$("schedule-title").value,prompt:$("schedule-prompt").value,timezone:$("schedule-timezone").value,[$("schedule-kind").value]:$("schedule-value").value});$("schedule-dialog").close();$("schedule-title").value="";$("schedule-prompt").value="";notice("Schedule created. You can pause it or inspect its runs here.");await showSchedules();}catch(error){$("schedule-dialog").close();notice(error.message,true);}finally{button.disabled=false;}};

async function describePausedRun(run) {
  const [questions, inbox] = await Promise.all([get("/v1/questions"), get("/v1/approvals")]);
  if (state.run?.id !== run.id || state.view !== "conversation") return;
  const question = questions.questions.find(item => item.session_id === run.session_id);
  const approval = inbox.approvals.some(item => item.session_id === run.session_id);
  const button = $("review-input");
  button.hidden = !question && !approval;
  if (question) {
    button.textContent = question.status === "pending" ? "Answer questions" : "Review saved answers";
    button.onclick = () => showQuestions().catch(error => notice(error.message, true));
    notice(question.status === "pending" ? "Harness needs your input. Answer the questions or skip them to continue." : "This question is ready to continue. Review its saved answers in Questions.");
  } else if (approval) {
    button.textContent = "Review approvals";
    button.onclick = () => showInbox().catch(error => notice(error.message, true));
    notice("This run has an action awaiting approval. Review it in Approvals, then resume the run.");
  } else {
    notice("This run is paused. Review its conversation and activity, then resume when ready.");
  }
}

function questionDraft(record, key) {
  if (!state.questionDrafts.has(record.id)) state.questionDrafts.set(record.id, {});
  const drafts = state.questionDrafts.get(record.id);
  if (!drafts[key]) drafts[key] = {choices: [], text: ""};
  return drafts[key];
}

function questionCard(record) {
  const answered = Object.keys(record.answers || {}).length;
  const item = card(`Conversation ${record.session_id.slice(-8)}`, `${answered} of ${record.questions.length} answers saved`, record.status);
  item.classList.add("question-card"); item.dataset.questionId = record.id;
  const form = element("form", undefined, "question-form");
  const pending = record.status === "pending";
  if (pending) item.append(element("p", `Answer by ${new Date(record.expires_at).toLocaleString()}. You can save some answers and return for the rest.`, "question-hint"));
  else if (record.status === "expired") item.append(element("p", "This question expired. Saved answers are kept; you can continue without answering the rest."));
  else if (record.status === "cancelled") item.append(element("p", "Unanswered questions were skipped. Saved answers are kept; continue when ready."));
  else item.append(element("p", "Your answers are saved. Continue the conversation when ready."));
  const inputs = [];
  record.questions.forEach((question, index) => {
    const key = `q${index}`;
    const field = element("fieldset", undefined, "question-field");
    field.append(element("legend", `${index + 1}. ${question.question}`));
    if (Object.prototype.hasOwnProperty.call(record.answers || {}, key)) {
      const value = record.answers[key];
      field.append(element("p", Array.isArray(value) ? value.join(" · ") || "No selections" : value || "No answer", "saved-answer"));
      field.append(element("p", "Saved · cannot be changed", "hint"));
      state.questionDrafts.get(record.id) && delete state.questionDrafts.get(record.id)[key];
    } else if (pending) {
      const draft = questionDraft(record, key);
      const multiple = question.multi_select && question.choices?.length;
      const group = element("div", undefined, "question-choices");
      const choices = [];
      const text = element("textarea"); text.id = `${record.id}-${key}-text`; text.rows = 2; text.maxLength = 8000; text.value = draft.text;
      text.dataset.questionKey = key;
      for (const [choiceIndex, choice] of (question.choices || []).entries()) {
        const label = element("label", undefined, "question-choice");
        const input = element("input"); input.type = multiple ? "checkbox" : "radio"; input.name = `${record.id}-${key}`; input.value = choice;
        input.checked = draft.choices.includes(choice); input.dataset.choiceIndex = choiceIndex;
        label.append(input, element("span", choice)); group.append(label); choices.push(input);
        input.onchange = () => { if (!multiple) {text.value = ""; draft.text = "";} draft.choices = choices.filter(option => option.checked).map(option => option.value); };
      }
      if (choices.length) { field.append(element("p", multiple ? "Choose any that apply, and add your own answer if needed." : "Choose one, or write your own answer.", "hint"), group); }
      const label = element("label", choices.length ? (multiple ? "Additional answer (optional)" : "Or write your answer") : "Your answer"); label.htmlFor = text.id;
      text.oninput = () => { draft.text = text.value; if (!multiple && text.value.trim()) {choices.forEach(input => {input.checked = false;}); draft.choices = [];} };
      field.append(label, text);
      inputs.push({key, multiple, text, choices});
    } else field.append(element("p", "No answer saved", "hint"));
    form.append(field);
  });
  const actions = element("div", undefined, "row-actions");
  if (pending) {
    const save = element("button", "Save answers", "primary"); save.type = "submit"; actions.append(save);
    actions.append(action("Skip unanswered questions", async () => {
      await post(`/v1/questions/${encodeURIComponent(record.id)}/cancel`);
      state.questionDrafts.delete(record.id);
      await showQuestions(); await refreshSidebar(); notice("Unanswered questions skipped. Saved answers are kept; you can continue the conversation.");
    }, "quiet"));
    form.onsubmit = async event => {
      event.preventDefault();
      const answers = {};
      for (const input of inputs) {
        const selected = input.choices.filter(choice => choice.checked).map(choice => choice.value);
        const text = input.text.value.trim();
        if (input.multiple) {const values = [...selected, ...(text ? [text] : [])]; if (values.length) answers[input.key] = values;}
        else if (text || selected.length) answers[input.key] = text || selected[0];
      }
      if (!Object.keys(answers).length) {notice("Choose or write at least one answer before saving.", true); return;}
      const controls = [...form.querySelectorAll("fieldset, button")];
      controls.forEach(control => {control.disabled = true;});
      try {
        const saved = await post(`/v1/questions/${encodeURIComponent(record.id)}/answer`, {answers});
        await showQuestions(); await refreshSidebar();
        notice(saved.status === "answered" ? "All answers saved. Continue the conversation when ready." : "Answers saved. You can return to finish the remaining questions.");
      } catch (error) {notice(error.message, true);} finally {controls.forEach(control => {control.disabled = false;});}
    };
    form.append(element("p", "Saved answers cannot be edited. Saving answers does not approve any actions.", "hint"));
  } else if (record.resume_run_id) {
    actions.append(action("Continue conversation", async () => {
      try {
        const run = await post(`/v1/runs/${encodeURIComponent(record.resume_run_id)}/resume`);
        await openSession(run.session_id);
      } catch (error) {
        notice(`Your saved answers are kept. ${error.message}\nReview any outstanding approvals or open the conversation before trying again.`, true);
      }
    }, "primary"));
    actions.append(action("Review approvals", showInbox, "quiet"));
  } else item.append(element("p", "Open the conversation to inspect its current run before continuing.", "hint"));
  actions.append(action("Open conversation", () => openSession(record.session_id), "quiet"));
  form.append(actions); item.append(form); return item;
}

async function showQuestions() {
  notice("");
  activate("questions", "Questions");
  $("list-toolbar").replaceChildren(element("p", "Answer the decisions and details Harness needs to continue your work. Saved answers remain available after you disconnect.", "muted"));
  const data = await get("/v1/questions");
  if (state.view !== "questions") return;
  state.questions = data.questions; $("question-count").textContent = data.questions.length;
  $("list-content").replaceChildren(...data.questions.map(questionCard));
  if (!data.questions.length) $("list-content").append(element("p", "There are no questions waiting for your input.", "empty-list"));
}
