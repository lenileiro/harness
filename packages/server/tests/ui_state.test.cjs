const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Node {
  constructor(tag='div') { this.tagName=tag; this.children=[]; this.className=''; this.value=''; this.disabled=false; this.hidden=false; this.textContent=''; this.files=[]; this.dataset={}; this.classList={add(){},remove(){},toggle(){}}; }
  append(...values) { this.children.push(...values); }
  replaceChildren(...values) { this.children=[...values]; }
  focus() {}
  querySelector() { return null; }
  querySelectorAll() { return []; }
  get childElementCount() {return this.children.length;}
}
const json = value => ({ok:true,json:async()=>value});
function deferred() {let resolve;const promise=new Promise(done=>{resolve=done});return {promise,resolve};}
function app(fetch) {
  const nodes=new Map();
  const document={getElementById(id){if(!nodes.has(id))nodes.set(id,new Node());return nodes.get(id);},createElement:tag=>new Node(tag),querySelectorAll:()=>[]};
  const context=vm.createContext({document,window:{},fetch:fetch|| (async url=>json(url==='/v1/health'?{user_id:'bob'}:url.startsWith('/v1/sessions')?{sessions:[]}:url==='/v1/approvals'?{approvals:[]}:{questions:[]})),DOMException,TextDecoder,AbortController,Uint8Array,URL,setTimeout,clearTimeout,btoa:input=>Buffer.from(input,'binary').toString('base64')});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/harness/server/static/app.js'),'utf8'),context);
  return {context,nodes,document,state:vm.runInContext('state',context)};
}
function file(name, read, size=4) {return {name,size,type:'text/plain',arrayBuffer:read};}
const bytes = () => new Uint8Array([115,97,102,101]).buffer;
const plain = value => JSON.parse(JSON.stringify(value));

test('pending file read cannot enter another authenticated identity',async()=>{
  const {context,document,state}=app(); state.token='alice';
  const pending=deferred();document.getElementById('files').files=[file('alice-private.txt',()=>pending.promise)];
  const upload=document.getElementById('files').onchange();
  document.getElementById('disconnect').onclick();
  await context.window.harnessConnect('bob');
  pending.resolve(bytes());await upload;
  assert.equal(state.token,'bob');assert.deepEqual(plain(state.attachments),[]);
  assert.equal(document.getElementById('attachments').children.length,0);
});

test('a rejected selection cannot hide an earlier file from the same selection',async()=>{
  const {document,state}=app();state.token='alice';
  document.getElementById('files').files=[file('small.txt',async()=>bytes()),file('too-large.txt',async()=>bytes(),9*1024*1024)];
  await document.getElementById('files').onchange();
  assert.deepEqual(plain(state.attachments),[]);
  assert.equal(document.getElementById('attachments').children.length,0);
  assert.match(document.getElementById('notice').textContent,/8 MiB/);
});

test('new conversation invalidates pending upload and preserves its empty composer',async()=>{
  const {context,document,state}=app();state.token='alice';state.session='A';
  const pending=deferred();document.getElementById('files').files=[file('a.txt',()=>pending.promise)];
  const upload=document.getElementById('files').onchange();context.newConversation();
  pending.resolve(bytes());await upload;
  assert.equal(state.session,null);assert.deepEqual(plain(state.attachments),[]);
});

test('submit stays disabled until selected file bytes are available',async()=>{
  const {document,state}=app();state.token='alice';const pending=deferred();
  document.getElementById('files').files=[file('ready.txt',()=>pending.promise)];
  const upload=document.getElementById('files').onchange();
  assert.equal(document.getElementById('send').disabled,true);
  pending.resolve(bytes());await upload;
  assert.equal(state.attachments.length,1);assert.equal(document.getElementById('send').disabled,false);
  assert.equal(document.getElementById('attachments').children.length,1);
});

test('late open-session response cannot replace the selected conversation run',async()=>{
  const pending=deferred();
  const {context,state}=app(async url=>{
    if(url==='/v1/runs?session_id=A') return await pending.promise;
    return json(url==='/v1/runs?session_id=B'?{runs:[]}:url.endsWith('/messages')?{messages:[]}:url==='/v1/sessions'?{sessions:[]}:url==='/v1/approvals'?{approvals:[]}:{questions:[]});
  });
  state.token='alice';const first=context.openSession('A');await context.openSession('B');
  pending.resolve(json({runs:[{id:'runA',session_id:'A',state:'completed'}]}));await first;
  assert.equal(state.session,'B');assert.equal(state.run,null);
});

test('late stream final status cannot resurrect a run after new conversation',async()=>{
  const pending=deferred();let statusRequested;const entered=new Promise(done=>{statusRequested=done});
  const {context,state}=app(async url=>{
    if(url.includes('/events'))return {ok:true,body:{getReader:()=>({read:async()=>({done:true})})}};
    if(url==='/v1/runs/runA'){statusRequested();return await pending.promise;}
    return json(url.endsWith('/messages')?{messages:[]}:url==='/v1/sessions'?{sessions:[]}:url==='/v1/approvals'?{approvals:[]}:{questions:[]});
  });
  state.token='alice';state.session='A';context.updateRun({id:'runA',session_id:'A',state:'running'});
  const stream=context.connectStream('runA');await entered;context.newConversation();
  pending.resolve(json({id:'runA',session_id:'A',state:'completed'}));await stream;
  assert.equal(state.session,null);assert.equal(state.run,null);
});

test('late submission stays in durable runs without navigating away from a new draft',async()=>{
  const pending=deferred();const {context,document,state}=app(async url=>url==='/v1/runs'?await pending.promise:json(url==='/v1/sessions'?{sessions:[]}:url==='/v1/approvals'?{approvals:[]}:{questions:[]}));
  state.token='alice';document.getElementById('prompt').value='first';
  const submit=document.getElementById('compose').onsubmit({preventDefault(){}});
  context.newConversation();document.getElementById('prompt').value='second draft';
  pending.resolve(json({id:'runA',session_id:'A',state:'queued'}));await submit;
  assert.equal(state.session,null);assert.equal(document.getElementById('prompt').value,'second draft');
});

test('failed file read preserves existing visible attachments and enables retry',async()=>{
  const {context,document,state}=app();state.token='alice';state.attachments=[{name:'kept.txt',kind:'file',mime_type:'text/plain',data:'a2VwdA=='}];context.renderAttachments();
  document.getElementById('files').files=[file('broken.txt',async()=>{throw new Error('Read failed')})];
  await document.getElementById('files').onchange();
  assert.equal(state.attachments.length,1);assert.equal(document.getElementById('attachments').children.length,1);
  assert.equal(document.getElementById('send').disabled,false);assert.match(document.getElementById('notice').textContent,/Read failed/);
});

test('failed stream exposes reconnect and keeps the current conversation',async()=>{
  const {context,document,state}=app(async()=>({ok:true,body:{getReader:()=>({read:async()=>{throw new Error('Connection interrupted')}})}}));
  state.token='alice';state.session='A';context.updateRun({id:'runA',session_id:'A',state:'running'});
  await context.connectStream('runA');
  assert.equal(state.session,'A');assert.equal(state.run.id,'runA');
  assert.equal(document.getElementById('reconnect-run').hidden,false);assert.match(document.getElementById('notice').textContent,/Connection interrupted/);
});

test('failed session lookup cannot leave controls targeting the previous conversation',async()=>{
  const {context,state}=app(async()=>{throw new Error('Offline')});state.token='alice';state.session='A';context.updateRun({id:'runA',session_id:'A',state:'running'});
  await assert.rejects(context.openSession('B'),/Offline/);
  assert.equal(state.session,'B');assert.equal(state.run,null);
});

test('late cancel response cannot target controls at a different conversation',async()=>{
  const pending=deferred();const {context,document,state}=app(async()=>await pending.promise);state.token='alice';state.session='A';context.updateRun({id:'runA',session_id:'A',state:'running'});
  const cancel=document.getElementById('cancel-run').onclick();context.newConversation();
  pending.resolve(json({id:'runA',session_id:'A',state:'cancelled'}));await cancel;
  assert.equal(state.session,null);assert.equal(state.run,null);
});
