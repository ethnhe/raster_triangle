

#ifndef RASTERTRIANGLE_H
#define RASTERTRIANGLE_H

/*

Enumerate pixels of a triangle within a rectangular region
Pixels on the boundary are considered inclusively -> connecting triangles may share pixels

Iterator usage:

PixelEnumerator<float> pixels(
	x0,y0,           //point 0, barycentric coordinate (0,0)
	x1,y1,           //point 1, barycentric coordinate (1,0)
	x2,y2,           //point 2, barycentric coordinate (0,1)
	x_lower_bound,   //for x<x_lower_bound or x>x_upper_bound (inclusive bounds), the pixel is clipped
	x_upper_bound,   //
	y_lower_bound,   //bounds for y
	y_upper_bound    //
);
int x,y;
float u,v;
while (pixels.getNext(&x,&y,&u,&v)){
	//(u,v) is the barycentric coordinate:
	//(x,y) = (x0,y0)+ (x1-x0,y1-y0)*u + (x2-x0,y2-y0)*v
	
	//blablabla
}

There is also a function allPixels that packs all generated pixels into a std::vector

*/

#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdio>

namespace RasterTriangle{

template<class Float>
struct PixelEnumerator{
	int lx,hx,ly,hy;
	Float x0,y0,x1,y1,x2,y2,u0,v0,u1,v1,u2,v2;
	int x,y,yhb;
	Float ubeg,vbeg,udelta,vdelta;
	Float ybeg;
	void sortPoints(){
		if (x0>x1){
			std::swap(x0,x1);
			std::swap(y0,y1);
			std::swap(u0,u1);
			std::swap(v0,v1);
		}
		if (x0>x2){
			std::swap(x0,x2);
			std::swap(y0,y2);
			std::swap(u0,u2);
			std::swap(v0,v2);
		}
		if (x1>x2){
			std::swap(x1,x2);
			std::swap(y1,y2);
			std::swap(u1,u2);
			std::swap(v1,v2);
		}
	}
	PixelEnumerator(Float x0,Float y0,Float x1,Float y1,Float x2,Float y2,int lx,int hx,int ly,int hy){
		this->x0=x0;
		this->y0=y0;
		this->u0=0;
		this->v0=0;
		this->x1=x1;
		this->y1=y1;
		this->u1=1;
		this->v1=0;
		this->x2=x2;
		this->y2=y2;
		this->u2=0;
		this->v2=1;
		this->lx=lx;
		this->hx=hx;
		this->ly=ly;
		this->hy=hy;
		sortPoints();
		x=int(std::ceil(std::max(Float(lx),std::min(Float(hx),this->x0))));
		//printf("%f %f %f\n",this->x0,this->x1,this->x2);
		//printf("%f %f %f\n",this->y0,this->y1,this->y2);
		if (x>=this->x0 && x<=this->x2){
			setUpY();
		}else{
			x=hx+1;
			yhb=hy;
			y=yhb+1;
			ybeg=0;
			ubeg=0;
			udelta=0;
			vbeg=0;
			vdelta=0;
		}
	}
	void setUpY(){
		Float yb,yc,ub,uc,vb,vc;
		if (x0==x2){
			if (y0<y1){
				yb=y0;
				ub=0;
				vb=0;
				yc=y1;
				uc=1;
				vc=0;
			}else{
				yc=y0;
				uc=0;
				vc=0;
				yb=y1;
				ub=1;
				vb=0;
			}
			if (y2<yb){
				yb=y2;
				ub=0;
				vb=1;
			}
			if (y2>yc){
				yc=y2;
				uc=0;
				vc=1;
			}
		}else{
			ub=0;
			vb=(Float(x)-x0)/(x2-x0);
			yb=y0+(y2-y0)*vb;
			if (x<=x1){
				if (x1!=x0){
					uc=(Float(x)-x0)/(x1-x0);
					vc=0;
					yc=y0+(y1-y0)*uc;
				}else{
					uc=1;
					vc=0;
					yc=y1;
				}
			}else{
				Float f=(Float(x)-x1)/(x2-x1);
				yc=y1+f*(y2-y1);
				uc=1-f;
				vc=f;
			}
			if (yb>yc){
				std::swap(yb,yc);
				std::swap(ub,uc);
				std::swap(vb,vc);
			}
		}
		y=int(std::ceil(std::max(Float(ly),std::min(Float(hy+1),yb))));
		yhb=int(std::floor(std::max(Float(ly-1),std::min(Float(hy),yc))));
		ybeg=yb;
		ubeg=ub;
		vbeg=vb;
		if (yb!=yc){
			udelta=(uc-ub)/(yc-yb);
			vdelta=(vc-vb)/(yc-yb);
		}else{
			udelta=0;
			vdelta=0;
		}
		//printf("setUpY x=%d y=%d %d u %f %f v %f %f ybeg %f\n",x,y,yhb,ubeg,udelta,vbeg,vdelta,ybeg);
	}
	bool getNext(int *x_out,int *y_out,Float *u,Float *v){
		
		while (y>yhb){
			x++;
			if (x>hx || x>std::floor(x2)){
				return false;
			}
			setUpY();
		}
		x_out[0]=x;
		y_out[0]=y;
		if (u || v){
			Float u_=ubeg+(Float(y)-ybeg)*udelta;
			Float v_=vbeg+(Float(y)-ybeg)*vdelta;
			if (u)
				u[0]=u0+(u1-u0)*u_+(u2-u0)*v_;
			if (v)
				v[0]=v0+(v1-v0)*u_+(v2-v0)*v_;
		}
		y++;
		return true;
	}
};

template<class Float>
std::vector<std::pair<std::pair<int,int>,std::pair<Float,Float> > > allPixels(Float x0,Float y0,Float x1,Float y1,Float x2,Float y2,int lx,int hx,int ly,int hy){
	PixelEnumerator<Float> pixels(x0,y0,x1,y1,x2,y2,lx,hx,ly,hy);
	int x,y;
	Float u,v;
	std::vector<std::pair<std::pair<int,int>,std::pair<Float,Float> > > ret;
	while (pixels.getNext(&x,&y,&u,&v)){
		ret.push_back(std::make_pair(
			std::make_pair(x,y),std::make_pair(u,v)));
	}
	return ret;
}

}//namespace

#endif
