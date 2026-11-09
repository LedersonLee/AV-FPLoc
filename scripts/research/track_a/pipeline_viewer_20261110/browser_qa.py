"""Headless integration test of the built viewer (no available CUA browser).
PYTHONPATH=<local playwright install> python browser_qa.py URL OUTPUT_DIR
"""
import json,sys
from pathlib import Path
from playwright.sync_api import sync_playwright
url=sys.argv[1];out=Path(sys.argv[2]);out.mkdir(exist_ok=True,parents=True)
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/home/jeongeonlee/.local/bin/google-chrome',headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1500,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(url);page.wait_for_function("document.querySelector('#status').textContent.startsWith('Ready')")
 initial=page.locator('#query').input_value();decision=page.locator('#decision').inner_text();heat=page.locator('#heatLeft').screenshot()
 page.locator('#yaw').select_option('7');assert heat!=page.locator('#heatLeft').screenshot()
 page.locator('#norm').select_option('raw');page.locator('#left').select_option('acoustic');page.locator('#right').select_option('fusion');page.locator('#shared').uncheck()
 page.locator('#gt').uncheck();assert page.locator('#decision').inner_text()==decision
 page.locator('#poses').uncheck();page.locator('#gt').check();page.locator('#poses').check()
 page.locator('#stage').select_option('visual');page.locator('#candidate').select_option(index=3);assert page.locator('#candidateDetail').inner_text().startswith('#'+page.locator('#candidate').input_value())
 page.locator('#rows button').nth(5).click();assert page.locator('#candidateDetail').inner_text().startswith('#'+page.locator('#candidate').input_value())
 page.locator('#stage').select_option('support')
 # Map click uses the documented candidate world coordinates and the viewer map transform.
 data=page.evaluate("""async()=>{const q=document.querySelector('#query').value;async function j(p){const r=await fetch(p);return JSON.parse(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).text())}return [await j(q+'.json.gz'),await j(q.split('/')[0]+'/bank.json.gz')]}""")
 q,b=data;s=q['seeds']['42'];candidate=next(x for x in s['candidates'] if x['id']==s['stages']['support'][4]);rect=page.locator('#backend').bounding_box();w=b['original_size'][0]*b['map_spacing_m'];h=b['original_size'][1]*b['map_spacing_m'];scale=min((900-44)/w,(900-44)/h);x=((900-w*scale)/2+(candidate['pose'][0]-b['world_bounds_m']['x_min'])*scale)*rect['width']/900;y=((900-h*scale)/2+(candidate['pose'][1]-b['world_bounds_m']['z_min'])*scale)*rect['height']/900
 page.locator('#backend').click(position={'x':x,'y':y});assert page.locator('#candidate').input_value()==str(candidate['id'])
 page.locator('#seed').select_option('43');page.wait_for_function("document.querySelector('#status').textContent.includes('seed 43')")
 page.locator('#filter').select_option('missing-top20');page.wait_for_function("document.querySelector('#status').textContent.startsWith('Ready') || document.querySelector('#status').textContent.startsWith('No sampled')")
 if page.locator('#query').is_enabled():assert 'missing-top20' in page.locator('#tags').inner_text()
 page.locator('#filter').select_option('all');page.wait_for_function("document.querySelector('#status').textContent.startsWith('Ready')")
 page.locator('#next').click();page.locator('#next').click();page.wait_for_function("document.querySelector('#status').textContent.startsWith('Ready') && document.querySelector('#status').textContent.includes(document.querySelector('#query').value)")
 page.locator('#left').select_option('visual');page.locator('#right').select_option('b2');page.locator('#yaw').select_option('max');page.locator('#norm').select_option('native');page.locator('#shared').check()
 page.screenshot(path=str(out/'desktop.png'),full_page=True)
 page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1');page.screenshot(path=str(out/'mobile.png'),full_page=True)
 assert not errors,errors
 (out/'browser_validation.json').write_text(json.dumps(dict(url=url,initial_query=initial,checks=['load gzip','yaw slice changes image','method/normalization/shared-range controls','GT overlay leaves final decision unchanged','candidate selector','candidate table click','candidate map click','seed43','failure filter','rapid next navigation consistency','desktop render','390px mobile no horizontal overflow'],console_errors=errors),indent=2)+'\n')
 browser.close()
print('BROWSER PASS',url)
