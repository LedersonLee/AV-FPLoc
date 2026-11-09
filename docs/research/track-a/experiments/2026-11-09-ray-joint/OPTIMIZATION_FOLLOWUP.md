# Bounded followup: interaction versus optimization/bottleneck

Primary40epochconcat trainloss~.175 vsattention~.077; attention−concat heldjoint+.586pp CI[−.553,+1.790] is not significant andattentiondoesnotbeatacoustic-only. A narrowflattenconcat14unitbottleneck can limitrepresentation despiteequalparametercounts; do notattributeanygap tointeractionalone.

Preserveprimarymodels. Trainnewconcat/attention/disabled at160epochs(1280updates),sameinputs/loss/seed/splits/optimizer. Also a widerconcat64unitbottleneck at40epochs(320updates),sameencodersandraydecoder, explicitlargerparametercount (~3×). This is a diagnosticstrongerconcatbaseline, notmatchedcapacity. Noheldearlystopping orwinnercheckpointselection. Comparelocalization andtrainloss; ifmoreupdates/morecapacity closesgap, originalgapwasnotattention-specificevidence. Noarchitecturegrid beyondthese4predeclaredarms.
