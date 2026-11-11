// Documented RLR C API only; no guessed internal ABI or opaque-context access.
#include "RLRAudioPropagation.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <iomanip>
void ck(RLRA_Error e){if(e!=RLRA_Success){std::cerr<<"API_ERROR "<<int(e)<<"\n";std::exit(3);}}
int main(int argc,char**argv){
 if(argc!=5)return 2;std::string out=argv[1],mode=argv[2];float d=std::atof(argv[3]);int repeats=std::atoi(argv[4]);
 RLRA_ContextConfiguration cfg={};cfg.thisSize=sizeof(cfg);ck(RLRA_ContextConfigurationDefault(&cfg));cfg.sampleRate=48000;cfg.indirectRayCount=20000;cfg.sourceRayCount=512;cfg.indirectRayDepth=50;cfg.sourceRayDepth=50;cfg.diffraction=true;cfg.transmission=false;cfg.threadCount=2;cfg.temporalCoherence=(mode=="temporal_on"||mode=="stationary_on");cfg.indirect=mode!="indirect_off";cfg.direct=mode!="direct_off";
 auto tick=std::chrono::steady_clock::now();RLRA_Context context=nullptr;ck(RLRA_CreateContext(&context,&cfg));ck(RLRA_AddSource(context));float s[3]={.17f,1.5f,d};ck(RLRA_SetSourcePosition(context,0,s));
 RLRA_ChannelLayout layout;layout.channelCount=1;layout.type=RLRA_ChannelLayoutType_Mono;
 int listeners=mode=="simultaneous"?6:1;
 for(int k=0;k<listeners;k++){ck(RLRA_AddListener(context,&layout));if(mode=="radius0")ck(RLRA_SetListenerRadius(context,k,0));}
 const float vertices[]={-12,-8,0,12,-8,0,12,12,0,-12,12,0,-12,-20,-12,12,-20,-12,12,-20,24,-12,-20,24};
 const uint32_t indices[]={0,1,2,0,2,3,2,1,0,3,2,0,4,6,5,4,7,6};ck(RLRA_AddMeshVertices(context,vertices,8));ck(RLRA_AddMeshIndices(context,indices,18,3,nullptr));ck(RLRA_AddObject(context));ck(RLRA_FinalizeObjectMesh(context,0));
 std::ofstream receipt(out+".json");receipt<<std::setprecision(10)<<"{\"mode\":\""<<mode<<"\",\"distance\":"<<d<<",\"sample_rate\":48000,\"unit_scale\":"<<cfg.unitScale<<",\"source_radius_default_m\":0,\"listener_radius_m\":"<<(mode=="radius0"?0:.1)<<",\"temporal_coherence\":"<<int(cfg.temporalCoherence)<<",\"direct\":"<<int(cfg.direct)<<",\"indirect\":"<<int(cfg.indirect)<<",\"records\":[";
 bool first=true;int stages=(mode=="simultaneous"||mode=="stationary_on"||mode=="stationary_off")?1:6;
 for(int rep=0;rep<repeats;rep++){
  for(int ch=0;ch<stages;ch++){
   for(int k=0;k<listeners;k++){int i=listeners==6?k:ch;double angle=M_PI+i*M_PI/3;float r[3]={s[0]+float(.05*cos(angle)),s[1],s[2]-float(.05*sin(angle))};ck(RLRA_SetListenerPosition(context,k,r));}
   ck(RLRA_Simulate(context));
   if(rep==0&&ch==0)ck(RLRA_WriteSceneMeshOBJ(context,(out+".obj").c_str()));
   for(int k=0;k<listeners;k++){
    int i=listeners==6?k:ch;size_t n=RLRA_GetIRSampleCount(context,k,0);const float* v=RLRA_GetIRChannel(context,k,0,0);if(!v||n==0)return 4;std::string file=out+"_r"+std::to_string(rep)+"_ch"+std::to_string(i)+".f32";std::ofstream wav(file,std::ios::binary);wav.write((const char*)v,n*sizeof(float));
    double angle=M_PI+i*M_PI/3;float rx=s[0]+float(.05*cos(angle)),rz=s[2]-float(.05*sin(angle));
    RLRA_Ray ray={};ray.origin[0]=s[0];ray.origin[1]=s[1];ray.origin[2]=s[2];ray.direction[0]=0;ray.direction[1]=0;ray.direction[2]=-1;ray.tMin=0;ray.tMax=100;ck(RLRA_TraceRayFirstHit(context,&ray));
    if(!first)receipt<<",";first=false;receipt<<"{\"repeat\":"<<rep<<",\"channel\":"<<i<<",\"samples\":"<<n<<",\"receiver\":["<<rx<<",1.5,"<<rz<<"],\"source\":["<<s[0]<<",1.5,"<<d<<"],\"ray_hit\":"<<int(ray.hit)<<",\"ray_distance\":"<<ray.tMax<<",\"normal\":["<<ray.normal[0]<<","<<ray.normal[1]<<","<<ray.normal[2]<<"]}";
   }
  }
 }
 receipt<<"],\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count()<<"}";ck(RLRA_DestroyContext(context));return 0;
}
