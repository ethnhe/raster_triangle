#include "rastertriangle.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;
using namespace cv;



extern "C"{

//n * 3 * 2 floats in xys
void rasterTrianglesRGB(int h,int w,unsigned char *img,int n,float * xys,unsigned char * colors){
	for (int i=0;i<n;i++){
		RasterTriangle::PixelEnumerator<float> pixels(
			xys[i*6+0],
			xys[i*6+1],
			xys[i*6+2],
			xys[i*6+3],
			xys[i*6+4],
			xys[i*6+5],
		0,h-1,0,w-1);
		//h/6,h-h/6-1,w/6,w-w/6-1);
		int x,y;
		float u,v;
		while (pixels.getNext(&x,&y,&u,&v)){
			img[(x*w+y)*3+0]=colors[i*9+0*3+0]*(1-u-v)+colors[i*9+1*3+0]*u+colors[i*9+2*3+0]*v;
			img[(x*w+y)*3+1]=colors[i*9+0*3+1]*(1-u-v)+colors[i*9+1*3+1]*u+colors[i*9+2*3+1]*v;
			img[(x*w+y)*3+2]=colors[i*9+0*3+2]*(1-u-v)+colors[i*9+1*3+2]*u+colors[i*9+2*3+2]*v;
		}
	}
}
void getAllPixels_phase1(int lx,int hx,int ly,int hy,float * xy3,void ** handle,int *n){
	std::vector<std::pair<std::pair<int,int>,std::pair<float,float> > > *all_pixels=new std::vector<std::pair<std::pair<int,int>,std::pair<float,float> > >(RasterTriangle::allPixels(
		xy3[0],xy3[1],xy3[2],xy3[3],xy3[4],xy3[5],lx,hx,ly,hy));
	handle[0]=all_pixels;
	n[0]=all_pixels->size();
}
void getAllPixels_phase2(void ** handle,int * xys,float * uvs){
	std::vector<std::pair<std::pair<int,int>,std::pair<float,float> > > *all_pixels=(std::vector<std::pair<std::pair<int,int>,std::pair<float,float> > >*)handle[0];
	int n=all_pixels->size();
	for (int i=0;i<n;i++){
		xys[i*2+0]=(*all_pixels)[i].first.first;
		xys[i*2+1]=(*all_pixels)[i].first.second;
		uvs[i*2+0]=(*all_pixels)[i].second.first;
		uvs[i*2+1]=(*all_pixels)[i].second.second;
	}
	delete all_pixels;
}


void getAllPixels_phase2_nouv(void ** handle,int * xys){
	std::vector<std::pair<std::pair<int,int>,std::pair<float,float> > > *all_pixels=(std::vector<std::pair<std::pair<int,int>,std::pair<float,float> > >*)handle[0];
	int n=all_pixels->size();
	for (int i=0;i<n;i++){
		xys[i*2+0]=(*all_pixels)[i].first.first;
		xys[i*2+1]=(*all_pixels)[i].first.second;
	}
	delete all_pixels;
}


int pixelsInTriangle(int h,int w,float * xy3,int *xys){
	void*  handle[1];
	int nret = 0;
 	getAllPixels_phase1(0,w-1,0,h-1,xy3,handle,&nret);	
	getAllPixels_phase2_nouv(handle,xys);
	return nret;
}


bool equal(float* points, int i,int j){
  	if ((abs(points[i*2] - points[j*2])<1e-5) && (abs(points[i*2+1] - points[j*2+1])<1e-5))
    		return 1;
  	return 0;

}
float dis(float x1,float y1,float x2,float y2){
  	return sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)  );
}

void swap(float&x,float&y){
  	float temp = x;  x =y;   y =temp;
}

float inter2(float * points_2d,float * cof,int ii,int jj,int i,int j){
  	float x1 = points_2d[i*2];
  	float y1 = points_2d[i*2+1];
  	float x2 = points_2d[j*2];
  	float y2 = points_2d[j*2+1];

  	float s1 = dis(ii,jj,x1,y1);
  	float s2 = dis(ii,jj,x2,y2);
  	return s2/(s1+s2)*cof[i] + s1/(s1+s2)*cof[j];
}

float inter(float* points_2d,float* cof,int ii,int jj,int i,int j,int k){
	if (equal(points_2d,i,j) &&  equal(points_2d,j,k) && equal(points_2d,i,k))
		return cof[i];
	if (equal(points_2d,i,j))
		return inter2(points_2d,cof,ii,jj,j,k);
	if (equal(points_2d,i,k))
		return inter2(points_2d,cof,ii,jj,j,k);
	if (equal(points_2d,j,k))
		return inter2(points_2d,cof,ii,jj,i,j);

	float x1 = points_2d[i*2];
	float y1 = points_2d[i*2+1];
	float x2 = points_2d[j*2];
	float y2 = points_2d[j*2+1];
	float x3 = points_2d[k*2];
	float y3 = points_2d[k*2+1];

	// in x_line or y_line
	if ((abs(y1-y2)<1e-5) && (abs(y3-y2)<1e-5))
		return inter2(points_2d,cof,ii,jj,i,j);
	if ((abs(x1-x2)<1e-5) && (abs(x3-x2)<1e-5))
		return inter2(points_2d,cof,ii,jj,i,j);

	// sort
	if (points_2d[i*2]>points_2d[j*2])
		swap(i,j);
	if (points_2d[j*2]>points_2d[k*2])
		swap(j,k);
	if (points_2d[i*2]>points_2d[j*2])
		swap(i,j);

	if ((ii>points_2d[j*2]) && (ii<points_2d[k*2]))
		swap(i,k);

	x1 = points_2d[i*2];y1 = points_2d[i*2+1];
	x2 = points_2d[j*2];y2 = points_2d[j*2+1];
	x3 = points_2d[k*2];y3 = points_2d[k*2+1];


	float ya =  y1 + (ii-x1)/(x2-x1)*(y2-y1);
	float yb =  y1 + (ii-x1)/(x3-x1)*(y3-y1);
	float cofa = (cof[i] * (ii - x2) + cof[j] * (x1 - ii)  ) / (x1-x2);
	float cofb = (cof[i] * (ii - x3) + cof[k] * (x1 - ii) ) / (x1-x3);
	float cof_ret = (cofa*(yb-jj) + cofb*(jj-ya) ) / (yb-ya);

	if ( cof_ret <min(cof[i],min(cof[j],cof[k])) )
		cof_ret = min(cof[i],min(cof[j],cof[k]));
	if ( cof_ret >max(cof[i],max(cof[j],cof[k])) )
		cof_ret = max(cof[i],max(cof[j],cof[k]));

	return cof_ret;

}


bool isline(float* xy3){
	float ax = xy3[0]-xy3[4];
	float ay = xy3[1]-xy3[5];
	float bx = xy3[2]-xy3[4];
	float by = xy3[3]-xy3[5];
	float cross = ax*by-bx*ay;
	if (abs(cross)<1e-5)
		return 1;
	return 0;

}


void cover_rgbd(float* p , float* pz, float* pr, float* pg, float* pb, int m1,int m2,int m3, int idx, int* triangle ,float* zbuf, int* rbuf, int* gbuf, int* bbuf, int h,int w, int * xys, int debug){
	float xy3[6];
	xy3[0] = p[m1*2+0];
	xy3[1] = p[m1*2+1];
	xy3[2] = p[m2*2+0];
	xy3[3] = p[m2*2+1];
	xy3[4] = p[m3*2+0];
	xy3[5] = p[m3*2+1];

	if (debug == 1){
		printf("%d, %d, %d\n", m1, m2, m3);
	}

	if (isline(xy3))
		return;

	int num = pixelsInTriangle(h,w,xy3,xys);
	for (int i =0; i<num; i++){
		int x = xys[i*2];
		int y = xys[i*2+1];
		float zz = inter(p,pz,x,y,m1,m2,m3);
		float r_f = inter(p,pr,x,y,m1,m2,m3);
		float g_f = inter(p,pg,x,y,m1,m2,m3);
		float b_f = inter(p,pb,x,y,m1,m2,m3);
		if (zz < zbuf[y*w+x]){
			zbuf[y*w+x] =  zz;
			rbuf[y*w+x] = int(r_f);
			gbuf[y*w+x] = int(g_f);
			bbuf[y*w+x] = int(b_f);
			triangle[y*w+x] = idx;
		}	
	}	
}

void rgbzbuffer(int h,int w, float* points_onface, float* points_onface_ori, float* points_z, float* points_r, float* points_g, float* points_b, int len_mesh, int* mesh, float* zbuf, int* rbuf, int* gbuf, int* bbuf){

	int * triangle = (int *) malloc(sizeof(int)*h*w);

	for (int i=0; i<h; i++)
		for (int j=0; j<w; j++){
			triangle[i*w+j] = -1, zbuf[i*w+j] = 1e9, rbuf[i*w+j] = gbuf[i*w+j] = bbuf[i*w+j] = 0;
		}
	
	int * xys = (int *) malloc(sizeof(int)*h*w*2);

	for (int i=0; i<len_mesh; i++){
		cover_rgbd(
			points_onface, points_z, points_r, points_g, points_b, mesh[i*3+0], 
			mesh[i*3+1], mesh[i*3+2], i, triangle, zbuf, rbuf, gbuf, bbuf,
			h, w, xys, 0 
		);
	}

	for (int i=0; i<h; i++)
		for (int j=0; j<w; j++){
			zbuf[i*w+j] = (zbuf[i*w+j] < 1e8) ? zbuf[i*w+j] : 0;
		}
	free(triangle);
	free(xys);
}


float bilinear_inter(int x1, int y1, int x2, int y2, float x, float y, float f11, float f21, float f12, float f22){
	float f_xy1=(x2-x)/(x2-x1)*f11+(x-x1)/(x2-x1)*f21;
	float f_xy2=(x2-x)/(x2-x1)*f12+(x-x1)/(x2-x1)*f22;
	float f=(y2-y)/(y2-y1)*f_xy1+(y-y1)/(y2-y1)*f_xy2;
	return f;
}

void cover_uvrgbd(
	float* p , float* pz, float* pu, float* pv, float* trgb, int m1,int m2,int m3, int idx,
	int* triangle, float* rgbzbuf, int h,int w, int th, int tw, int * xys, int debug
){

	float xy3[6];
	xy3[0] = p[m1*2+0];
	xy3[1] = p[m1*2+1];
	xy3[2] = p[m2*2+0];
	xy3[3] = p[m2*2+1];
	xy3[4] = p[m3*2+0];
	xy3[5] = p[m3*2+1];

	if (debug == 1){
		printf("%d, %d, %d\n", m1, m2, m3);
	}

	if (isline(xy3))
		return;

	int num = pixelsInTriangle(h,w,xy3,xys);
	for (int i =0; i<num; i++){
		int x = xys[i*2];
		int y = xys[i*2+1];
		float zz = inter(p,pz,x,y,m1,m2,m3);
		float uu = inter(p,pu,x,y,m1,m2,m3);
		float vv = inter(p,pv,x,y,m1,m2,m3);
		if (zz < rgbzbuf[y*w*4+x*4+3]){
			float tx = (uu * tw) - 0.5;
    		float ty = (1. - vv) * th - 0.5;
			int x1=int(tx);
			int x2=int(tx)+1;
			int y1=int(ty);
			int y2=int(ty)+1;
			if ((x2>=tw)|(y2>=th)){
				rgbzbuf[y*w*4+x*4+0] = trgb[y1*tw*3+x1*3+0];
				rgbzbuf[y*w*4+x*4+1] = trgb[y1*tw*3+x1*3+1];
				rgbzbuf[y*w*4+x*4+2] = trgb[y1*tw*3+x1*3+2];
			}else{
				rgbzbuf[y*w*4+x*4+0] = bilinear_inter(
					x1, y1, x2, y2, tx, ty, trgb[y1*tw*3+x1*3+0],
					trgb[y1*tw*3+x2*3+0], trgb[y2*tw*3+x1*3+0],
					trgb[y2*tw*3+x2*3+0]
				);
				rgbzbuf[y*w*4+x*4+1] = bilinear_inter(
					x1, y1, x2, y2, tx, ty, trgb[y1*tw*3+x1*3+1],
					trgb[y1*tw*3+x2*3+1], trgb[y2*tw*3+x1*3+1],
					trgb[y2*tw*3+x2*3+1]
				);
				rgbzbuf[y*w*4+x*4+2] = bilinear_inter(
					x1, y1, x2, y2, tx, ty, trgb[y1*tw*3+x1*3+2],
					trgb[y1*tw*3+x2*3+2], trgb[y2*tw*3+x1*3+2],
					trgb[y2*tw*3+x2*3+2]
				);
			}
			rgbzbuf[y*w*4+x*4+3] = zz;
			triangle[y*w+x] = idx;
		}	
	}	
}

void uv_rgbzbuffer(
	int h,int w, float* points_onface, float* points_onface_ori, float* points_z, float* points_u, float* points_v, int len_mesh, int* mesh,
	int th, int tw, float* tmap, float* rgbzbuf
){

	int * triangle = (int *) malloc(sizeof(int)*h*w);

	for (int i=0; i<h; i++)
		for (int j=0; j<w; j++){
			triangle[i*w+j] = -1,
			rgbzbuf[i*w*4+j*4+3] = 1e9, rgbzbuf[i*w*4+j*4+0] = rgbzbuf[i*w*4+j*4+1] = rgbzbuf[i*w*4+j*4+2] = 0;
		}
	
	int * xys = (int *) malloc(sizeof(int)*h*w*2);

	for (int i=0; i<len_mesh; i++){
		cover_uvrgbd(
			points_onface, points_z, points_u, points_v, tmap, mesh[i*3+0], 
			mesh[i*3+1], mesh[i*3+2], i, triangle, rgbzbuf,
			h, w, th, tw, xys, 0 
		);
	}

	for (int i=0; i<h; i++)
		for (int j=0; j<w; j++){
			rgbzbuf[i*w*4+j*4+3] = (rgbzbuf[i*w*4+j*4+3] < 1e8) ? rgbzbuf[i*w*4+j*4+3] : 0;
		}
	free(triangle);
	free(xys);
}


void cover_d(float* p , float* pz, int m1,int m2,int m3, int idx, int* triangle ,float* zbuf, int h,int w, int * xys, int debug){
	float xy3[6];
	xy3[0] = p[m1*2+0];
	xy3[1] = p[m1*2+1];
	xy3[2] = p[m2*2+0];
	xy3[3] = p[m2*2+1];
	xy3[4] = p[m3*2+0];
	xy3[5] = p[m3*2+1];

	if (debug == 1){
		printf("%d, %d, %d\n", m1, m2, m3);
	}

	if (isline(xy3))
		return;

	int num = pixelsInTriangle(h,w,xy3,xys);
	for (int i =0; i<num; i++){
		int x = xys[i*2];
		int y = xys[i*2+1];
		float zz = inter(p,pz,x,y,m1,m2,m3);
		if (zz < zbuf[y*w+x]){
			zbuf[y*w+x] =  zz;
			triangle[y*w+x] = idx;
		}	
	}	
}

void zbuffer(int h,int w, float* points_onface, float* points_z, int len_mesh, int* mesh, float* zbuf){
	int * triangle = (int *) malloc(sizeof(int)*h*w);	

	for (int i=0; i<h; i++)
		for (int j=0; j<w; j++){
			triangle[i*w+j] = -1, zbuf[i*w+j] = 1e9;
		}
	
	int * xys = (int *) malloc(sizeof(int)*h*w*2);

	for (int i=0; i<len_mesh; i++){
		cover_d(
			points_onface, points_z, mesh[i*3+0], 
			mesh[i*3+1], mesh[i*3+2], i, triangle, zbuf,
			h, w, xys, 0 
		);
	}

	for (int i=0; i<h; i++)
		for (int j=0; j<w; j++){
			zbuf[i*w+j] = (zbuf[i*w+j] < 1e8) ? zbuf[i*w+j] : 0;
		}
	free(triangle);
	free(xys);
}
	

void pcld_zbuffer(int h, int w, int* points_onface, float* points_z, int len_pts, int* blur_directions, int n_blur, float* zbuf){
        /*
         * h, w: rendered image height and width
         * points_onface: flatten uv: (p3d[:, :2].dot(K.T) / p3d[:, 2:] + 0.5).astype(np.int32).flatten()
         * points_z: p3ds[:, 2]
         * len_pts: number of points to be rendered
         * blur_directions: bluring direction of each point on 2D depth, each rendered depth value will diffuse to adjacent pixels. eg: [(-1, -1), (-1, 0), (-1, 1), ..., (1, 1)] for 9 adjacent pixels
         * n_blur: number of blur directions
         * zbuf: output zbuffer
         */
        for (int i=0; i<h; i++)
                for (int j=0; j<w; j++){
                        zbuf[i*w+j] = 1e9;
                }
        
        for (int i=0; i<len_pts; i++){
                int u=points_onface[i*2], v=points_onface[i*2+1];
                // printf("ori_uv: %d, %d\n", u, v);
                for (int j=0; j<n_blur; j++){
                        int bu=blur_directions[j*2], bv=blur_directions[j*2+1];
                        u=u+bu, v=v+bv;
                        u=max(u, 0), v=max(v, 0);
                        u=min(u, w-1), v=min(v, h-1);
                        // printf("uv_cng: %d, %d\n", u, v);
                        // printf("zbuf_v, z: %f %f\n", zbuf[v*w+u], points_z[i]);
                        zbuf[v*w+u] = min(zbuf[v*w+u], points_z[i]);
                }
        }

        for (int i=0; i<h; i++)
                for (int j=0; j<w; j++){
                        zbuf[i*w+j] = (zbuf[i*w+j] < 1e8) ? zbuf[i*w+j] : 0;
                }
}


}//extern "C"
