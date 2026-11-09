import sys,json,numpy,sklearn,torch,scipy,platform
from pathlib import Path
Path('/output/environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),numpy=numpy.__version__,sklearn=sklearn.__version__,torch=torch.__version__,scipy=scipy.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),docker_image_id='sha256:e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318'),indent=2))
