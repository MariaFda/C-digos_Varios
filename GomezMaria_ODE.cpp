
#include <stdio.h>
#include <cmath>

#define max 101 
#define max1 3000                    

#define g 10 //unidades 
#define c 0.2
#define m 0.3 
#define velocidadInicial 300.0
#define t_maximo 300
#define pasos 1000


int aceleracion_x(int v_x, int v_y){
    float magnitud=sqrt(v_x*v_x+v_y*v_y);
    float a_x=-(c/m)*magnitud*magnitud;

    return a_x;

}

int aceleracion_y(int v_x, int v_y){
    float magnitud=sqrt(v_x*v_x+v_y*v_y);
    float a_y=-g-(c/m)*magnitud*magnitud;
    return a_y;

}
int rungeVelocidad_x(int v_x, int v_y){
float h=t_maximo/pasos; 
float l1_x;
float k1_x;
float l1_y;
float k1_y;
float l2_x;
float k2_x;
float l2_y;
float k2_y;
float l3_x;
float k3_x;
float l3_y;
float k3_y;
float l4_x;
float k4_x;
float l4_y;
float k4_y;
float v_nx;
    
    l1_x=h*aceleracion_x(v_x,v_y);
    k1_x=h*(v_x);
    
    l1_y=h*aceleracion_y(v_x,v_y);
    k1_y=h*(v_y);
    
    l2_x=h*aceleracion_x(v_x,v_y+0.5*l1_y);
    k2_x=h*(v_x+0.5*l2_x);
    
    l2_y=h*aceleracion_y(v_x+0.5*l1_x,v_y);
    k2_y=h*(v_y+0.5*l2_y);
    
    l3_x=h*aceleracion_x(v_x,v_y+0.5*l2_y);
    k3_x=h*(v_x+0.5*l3_x);
    
    l3_y=h*aceleracion_y(v_x+0.5*l2_x,v_y);
    k3_y=h*(v_y+0.5*l3_y);
    
    l4_x=h*aceleracion_x(v_x,v_y+0.5*l3_y);
    k4_x=h*(v_x+0.5*l4_x);
    
    l4_y=h*aceleracion_y(v_x+0.5*l3_x,v_y);
    k4_y=h*(v_y+0.5*l4_y);
    
    v_nx=v_x+(1/6)*(l1_x+2*l2_x+2*l3_x+l4_x);
    
    return v_nx;

}
int rungeVelocidad_y(int v_x,int v_y){
float h=t_maximo/pasos; 
float l1_x;
float k1_x;
float l1_y;
float k1_y;
float l2_x;
float k2_x;
float l2_y;
float k2_y;
float l3_x;
float k3_x;
float l3_y;
float k3_y;
float l4_x;
float k4_x;
float l4_y;
float k4_y;
float v_ny;
    
    l1_x=h*aceleracion_x(v_x,v_y);
    k1_x=h*(v_x);
    
    l1_y=h*aceleracion_y(v_x,v_y);
    k1_y=h*(v_y);
    
    l2_x=h*aceleracion_x(v_x,v_y+0.5*l1_y);
    k2_x=h*(v_x+0.5*l2_x);
    
    l2_y=h*aceleracion_y(v_x+0.5*l1_x,v_y);
    k2_y=h*(v_y+0.5*l2_y);
    
    l3_x=h*aceleracion_x(v_x,v_y+0.5*l2_y);
    k3_x=h*(v_x+0.5*l3_x);
    
    l3_y=h*aceleracion_y(v_x+0.5*l2_x,v_y);
    k3_y=h*(v_y+0.5*l3_y);
    
    l4_x=h*aceleracion_x(v_x,v_y+0.5*l3_y);
    k4_x=h*(v_x+0.5*l4_x);
    
    l4_y=h*aceleracion_y(v_x+0.5*l3_x,v_y);
    k4_y=h*(v_y+0.5*l4_y);
    
    v_ny=v_y+(1/6)*(l1_y+2*l2_y+2*l3_y+l4_y);
    
    return v_ny;
}
int main()
{


  return 0;
}
 
