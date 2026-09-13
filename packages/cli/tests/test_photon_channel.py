import asyncio
import json
import shutil
from pathlib import Path

import pytest

from harness.cli.channels.photon import BRIDGE, PhotonTransport
from harness.cli.channels.transports import ChannelError
from harness.core.gateway_channels import ChannelConfig, ChannelStore


@pytest.mark.skipif(shutil.which("node") is None, reason="Photon SDK bridge requires Node")
async def test_actual_node_photon_sdk_contract_scoped_dispatch_send_and_shutdown(tmp_path):
    shim = tmp_path / "sdk.mjs"
    shim.write_text(
        "import {runBridge} from " + json.dumps(BRIDGE.as_uri()) + ";\n"
        'if(process.env.UNRELATED_SECRET) throw new Error("environment leaked");\n'
        'if(process.env.PHOTON_PROJECT_SECRET !== "project-secret") throw new Error("wrong credentials");\n'
        'const space = {id:"space",type:"dm", async send(builder){if(builder.text !== "Reviewed reply 😀") throw new Error("text changed");return {id:"sent"}}};\n'
        "const app={messages:(async function*(){\n"
        'for(const sender of ["owner","owner","stranger"]) yield [space,{id:"in-"+sender,direction:"inbound",sender:{id:sender},content:{type:"text",text:"approve pending"}}];\n'
        'yield [space,{id:"self",direction:"outbound",sender:{id:"owner"},content:{type:"text",text:"ignore"}}];\n'
        "await new Promise(()=>{});})()};\n"
        'process.stdout.write(JSON.stringify({event:"ready",project:process.env.PHOTON_PROJECT_ID})+"\\n");\n'
        'await runBridge({app,imessage:()=>({space:{get:async id=>{if(id!=="space") throw new Error("wrong space");return space}}}),text:value=>({text:value}),input:process.stdin,output:process.stdout});\n'
    )
    processes = []

    async def factory(command, script, **kwargs):
        assert Path(script) == BRIDGE
        process = await asyncio.create_subprocess_exec(command, str(shim), **kwargs)
        processes.append(process)
        return process

    transport = PhotonTransport(
        config=ChannelConfig(app_id="project", allowed_users=["owner"]),
        token="project-secret",
        process_factory=factory,
    )
    store = ChannelStore(cwd=tmp_path, transport="photon")
    try:
        await transport.authenticate()
        task = asyncio.create_task(transport.receive(store))
        for _ in range(100):
            if store.status()["inbox"]:
                break
            await asyncio.sleep(0.01)
        message = store.claim_message()
        assert message and message.user_id == "owner" and message.thread_id == '["project","space"]'
        assert store.claim_message() is None
        await transport.send(message.thread_id, "Reviewed reply 😀", "delivery")
        with pytest.raises(ChannelError, match="different project"):
            await transport.send('["other","space"]', "private", "id")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    finally:
        await transport.close()
        store.close()
    assert processes and all(process.returncode is not None for process in processes)
