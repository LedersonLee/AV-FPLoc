#include "RLRAudioPropagation.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>
void ck(RLRA_Error e){if(e!=RLRA_Success){std::cerr<<"API_ERROR "<<int(e)<<"\n";std::exit(3);}}
int main(int argc,char**argv){
 if(argc!=9)return 2;std::string prefix=argv[1],mesh=argv[2],material=argv[3],mode=argv[7];int audit=std::atoi(argv[8]);float source[3]={float(std::atof(argv[4])),float(std::atof(argv[5])),float(std::atof(argv[6]))};
 RLRA_ContextConfiguration cfg={};cfg.thisSize=sizeof(cfg);ck(RLRA_ContextConfigurationDefault(&cfg));cfg.sampleRate=48000;cfg.indirectRayCount=20000;cfg.sourceRayCount=512;cfg.indirectRayDepth=50;cfg.sourceRayDepth=50;cfg.threadCount=2;cfg.transmission=false;cfg.diffraction=true;cfg.temporalCoherence=false;cfg.direct=mode!="indirect_only";cfg.indirect=mode!="direct_only";
 auto start=std::chrono::steady_clock::now();RLRA_Context ctx=nullptr;ck(RLRA_CreateContext(&ctx,&cfg));ck(RLRA_SetMaterialDatabaseJSON(ctx,material.c_str()));ck(RLRA_AddSource(ctx));ck(RLRA_SetSourcePosition(ctx,0,source));RLRA_ChannelLayout layout;layout.channelCount=1;layout.type=RLRA_ChannelLayoutType_Mono;ck(RLRA_AddListener(ctx,&layout));
 std::ifstream in(mesh);size_t nv,nf;in>>nv>>nf;std::vector<float> vertices(3*nv);for(auto&v:vertices)in>>v;std::vector<uint32_t> varied,background;
 for(size_t f=0;f<nf;f++){uint32_t x,y,z;int label;in>>x>>y>>z>>label;auto&v=label?background:varied;v.insert(v.end(),{x,y,z});}if(!in)return 5;
 ck(RLRA_AddMeshVertices(ctx,vertices.data(),nv));if(varied.size())ck(RLRA_AddMeshIndices(ctx,varied.data(),varied.size(),3,"audit_surface"));if(background.size())ck(RLRA_AddMeshIndices(ctx,background.data(),background.size(),3,"fixed_background"));ck(RLRA_AddObject(ctx));ck(RLRA_FinalizeObjectMesh(ctx,0));
 std::ofstream receipt(prefix+".json");receipt<<std::setprecision(12)<<"{\"frequency_bands\":"<<cfg.frequencyBands<<",\"sample_rate\":"<<cfg.sampleRate<<",\"max_ir_length_s\":"<<cfg.maxIRLength<<",\"unit_scale\":"<<cfg.unitScale<<",\"global_volume\":"<<cfg.globalVolume<<",\"direct_rays\":"<<cfg.directRayCount<<",\"listener_radius_m\":0.1,\"lengths\":[";
 for(int c=0;c<6;c++){double angle=M_PI+c*M_PI/3;float r[3]={source[0]+float(.05*cos(angle)),source[1],source[2]-float(.05*sin(angle))};ck(RLRA_SetListenerPosition(ctx,0,r));ck(RLRA_Simulate(ctx));if(c==0&&audit){ck(RLRA_WriteSceneMeshOBJ(ctx,(prefix+".obj").c_str()));ck(RLRA_WriteIRMetrics(ctx,0,0,(prefix+"_metrics.txt").c_str()));}size_t n=RLRA_GetIRSampleCount(ctx,0,0);auto v=RLRA_GetIRChannel(ctx,0,0,0);if(!v||!n)return 4;std::ofstream out(prefix+"_ch"+std::to_string(c)+".f32",std::ios::binary);out.write((const char*)v,n*sizeof(float));if(c)receipt<<",";receipt<<n;}
 receipt<<"],\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}";ck(RLRA_DestroyContext(ctx));return 0;
}
