from curses import COLOR_RED
import taichi as ti
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu,device_memory_fraction=0.3)

ImgSize=512
ClothWid=4.0
ClothHgt=4.0
ClothResX=35
step=1024

scalar = lambda: ti.field(dtype=ti.f32)
vec = lambda: ti.Vector.field(3, dtype=ti.f32)

Pos_Pre=vec()
pos=vec()
vel=vec()
F=vec()
acc=vec()

KStruct=scalar()
KShear=scalar()
KBend=scalar()
loss_n=scalar()
mass=scalar()
airdamping=scalar()
z=scalar()

ti.root.dense(ti.ijk,(step,ClothResX+1,ClothResX+1)).place(Pos_Pre,pos,vel,F,acc)
ti.root.dense(ti.i,2).place(KStruct,KShear,KBend)
ti.root.place(mass,airdamping,loss_n,z)
ti.root.lazy_grad()

gravity=ti.Vector([0.0,-0.05,0.0])
# gravity=ti.Vector([0.0,-9.8,0.0])
a=-0.5
b=2.0
dt=0.05
learning_rate=1.0

KStruct[0]=50.0
KShear[0]=50.0
KBend[0]=50.0
KStruct[1]=0.25
KShear[1]=0.25
KBend[1]=0.25
# KStruct[0]=500.0
# KShear[0]=500.0
# KBend[0]=500.0
# KStruct[1]=0.025
# KShear[1]=0.025
# KBend[1]=0.025

mass[None]=1.0
# mass[None]=0.1
airdamping[None]=0.0125
# airdamping[None]=0.0125
loss_n[None]=0.0
z[None]=-ClothHgt/2.0

#########################
N_Triangles=ClothResX*ClothResX*2
indices=ti.field(dtype=ti.i32,shape=N_Triangles*3)
vertices=ti.Vector.field(3,dtype=ti.f32,shape=(ClothResX+1)*(ClothResX+1))
##########################

@ti.func
def Get_X(n:ti.i32) ->ti.i32:
    ax=0
    if (n==0) or (n==4) or (n==7):
        ax=1
    elif (n==2) or (n==5) or (n==6):
        ax=-1
    elif (n==8):
        ax=2
    elif (n==10):
        ax=-2
    else:
        ax=0
    return ax

@ti.func
def Get_Y(n:ti.i32)->ti.i32:
    ax=0
    if (n==1) or (n==4) or (n==5):
        ax=-1
    elif (n==3) or (n==6) or (n==7):
        ax=1
    elif (n==9):
        ax=-2
    elif (n==11):
        ax=2
    else:
        ax = 0
    return ax

@ti.func
def Compute_Force(t:ti.i32,i:ti.i32,j:ti.i32):
    p1=pos[t,i,j]
    F[t,i,j]=gravity*mass[None]-vel[t,i,j]*airdamping[None]
    v1=vel[t,i,j]

    for n in range(0,12): 
        Spring_Type=ti.Vector([Get_X(n),Get_Y(n)]) 
        Coord_Neigh=Spring_Type + ti.Vector([i,j]) 
        if (Coord_Neigh.x>=0) and (Coord_Neigh.x<=ClothResX) and (Coord_Neigh.y>=0) and (Coord_Neigh.y<=ClothResX): 
            Sping_Vector=Spring_Type*ti.Vector([ClothWid/ClothResX,ClothHgt/ClothResX])
            Rest_Length=ti.sqrt(Sping_Vector.x**2+Sping_Vector.y**2) 
            p2=pos[t,Coord_Neigh.x,Coord_Neigh.y]
            v2=(p2-Pos_Pre[t,Coord_Neigh.x,Coord_Neigh.y])/dt
            deltaV=v1-v2
            deltaP=p1-p2
            dist=ti.sqrt(deltaP.x**2+deltaP.y**2+deltaP.z**2)
            Sping_Force=ti.Vector([0.0,0.0,0.0])
            AX = 0.0
            BX = 0.0
            if (n<4):
            # AX=(-KStruct[0]*(dist-Rest_Length)*((deltaP)/dist))
                AX=-KStruct[0]*(dist-Rest_Length)
                BX=-KStruct[1]*(deltaV.dot(deltaP)/dist)
            elif (n>=4 and n<8):
                AX=-KShear[0]*(dist-Rest_Length)
                BX=-KShear[1]*(deltaV.dot(deltaP)/dist)
            else:
                AX=-KBend[0]*(dist-Rest_Length)
                BX=-KBend[1]*(deltaV.dot(deltaP)/dist)
            Sping_Force = deltaP.normalized()*(AX+BX)
            
            # if (n<4):
            #     Sping_Force=(-KStruct[0]*(dist-Rest_Length)*((deltaP)/dist))
            # elif (n>=4 and n<8):
            #     Sping_Force=(-KShear[0]*(dist-Rest_Length)*((deltaP)/dist))
            # else:
            #     Sping_Force=(-KBend[0]*(dist-Rest_Length)*((deltaP)/dist))

            F[t,i,j]+=Sping_Force
    
@ti.func
def collision(t:ti.i32,i:ti.i32,j:ti.i32): 
    if(pos[t,i,j].y<0):
        pos[t,i,j].y=0 

@ti.kernel
def Reset_Cloth(): 
    for t,i,j in pos:
        pos[t,i,j] = ti.Vector([ClothWid*(i/ClothResX)-ClothWid/2.,0.0,ClothHgt*(j/ClothResX)-ClothHgt/2.0])
        Pos_Pre[t,i,j]=pos[t,i,j]
        vel[t,i,j]=ti.Vector([0.0,0.0,0.0])
        F[t,i,j]=ti.Vector([0.0,0.0,0.0])
        acc[t,i,j]=ti.Vector([0.0,0.0,0.0]) 

#########################
        if i<ClothResX-1 and j<ClothResX-1:
            Tri_Id=((i*(ClothResX-1))+j)*2
            indices[Tri_Id*3+2]=i*ClothResX+j
            indices[Tri_Id*3+1]=(i+1)*ClothResX+j
            indices[Tri_Id*3+0]=i*ClothResX+(j+1)

            Tri_Id+=1
            indices[Tri_Id*3+2]=(i+1)*ClothResX+j+1
            indices[Tri_Id*3+1]=i*ClothResX+(j+1)
            indices[Tri_Id*3+0]=(i+1)*ClothResX+j
##########################

@ti.kernel
def simulation(t:ti.i32):
    for i in range(step):
        for j in range(step):
            Compute_Force(t,i,j)

            acc[t,i,j]=F[t,i,j]/mass[None]
            tmp=pos[t,i,j]
            pos[t,i,j]=pos[t,i,j]*2.0-Pos_Pre[t,i,j]+acc[t,i,j]*dt*dt
            vel[t,i,j]=(pos[t,i,j]-Pos_Pre[t,i,j])/dt
            Pos_Pre[t,i,j]=tmp

            # acc[coord]=F[coord]/mass[None]
            # vel[coord]+=acc[coord]*dt
            # pos[coord]+=vel[coord]*dt

            if ((i == 0 and j == 0 ) or (i == 35 and j == 0 )): 
                if z[None] <= ClothHgt/2.0:
                    z[None] += (ClothWid/400)
                else:
                    z[None] = 2.0 
                y=a*z[None]**2+b
                x=0.0
                if pos[t,i,j].x <0.0:
                    x=-ClothHgt/2.0
                else:
                    x=ClothHgt/2.0
                pos[t,i,j]=ti.Vector([x,y,z[None]])
            # else:                   
                # acc[coord]=F[coord]/mass[None]
                # tmp=pos[coord]
                # pos[coord]=pos[coord]*2.0-Pos_Pre[coord]+acc[coord]*dt*dt
                # vel[coord]=(pos[coord]-Pos_Pre[coord])/dt
                # Pos_Pre[coord]=tmp

                # Pos_Pre[coord]=pos[coord]
                # acc[coord]=F[coord]/mass[None]
                # vel[coord]+=acc[coord]*dt
                # pos[coord]+=vel[coord]*dt
            collision(t,i,j) 

@ti.kernel
def Compute_Loss(t:ti.i32):
    current=pos[t,18,0]
    target=ti.Vector([0.0,0.0,2.0])
    Loss_Vector=current-target
    loss_n[None]=0.5*ti.sqrt(Loss_Vector.x**2+Loss_Vector.y**2+Loss_Vector.z**2) 

@ti.kernel
def Vec_Clear(): 
    for t,i,j in pos:
        pos.grad[t,i,j]=ti.Vector([0.0,0.0,0.0])
        vel.grad[t,i,j]=ti.Vector([0.0,0.0,0.0])
        F.grad[t,i,j]=ti.Vector([0.0,0.0,0.0])
        acc.grad[t,i,j]=ti.Vector([0.0,0.0,0.0]) 

@ti.kernel
def Scalar_Clear(): 
    KStruct.grad[0]=0.0
    KShear.grad[0]=0.0
    KBend.grad[0]=0.0
    KStruct.grad[1]=0.0
    KShear.grad[1]=0.0
    KBend.grad[1]=0.0
    loss_n.grad[None]=0.0
    mass.grad[None]=0.0
    airdamping.grad[None]=0.0
    z.grad[None]=0.0

def Grad_Clear(): 
    Vec_Clear()
    Scalar_Clear()

def forward():
    for t in range(step):
        simulation(t)
    Compute_Loss(step-1)

##########################################
@ti.kernel
def update_verts():
    for i,j in ti.ndrange(ClothResX, ClothResX):
        vertices[i*ClothResX+j]=pos[i,j]
##########################################


def main():
    Reset_Cloth()
    forward()
    Grad_Clear()
    Reset_Cloth()
    losses=[]
    for i in range(step):
        with ti.ad.Tape(loss=loss_n,clear_gradients=True):
            forward()
        print("i=",i,"loss=",loss_n[None])
        losses.append(loss_n[None])
        for i in range(2):
            KStruct[i]-=learning_rate * KStruct.grad[i]
            KShear[i]-=learning_rate * KShear.grad[i]
            KBend[i]-=learning_rate * KBend.grad[i]
    for i in range(2):
        print("KStruct",KStruct[0])
        print(KStruct[1])
        print("KShear",KShear[0])
        print(KShear[1])
        print("KBend",KBend[0])
        print(KBend[1])
    
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    plt.plot(losses)
    plt.title("loss_change")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()



#####################################
# Reset_Cloth()
# ggui=ti.ui.Window('Cloth',(ImgSize,ImgSize),vsync=True)
# canvas=ggui.get_canvas()
# scene=ti.ui.Scene()
# camera=ti.ui.Camera()
# canvas.set_background_color((1, 1, 1))
# camera.position(5.0,3.0,5.0)
# camera.lookat(0.0,0.0,0.0)
# while ggui.running:
#     simulation()
#     update_verts()
#     scene.mesh(vertices,indices=indices,color=(1.,1.,1.))
#     scene.point_light(pos=(10.0,10.0,0.0),color=(1.0,1.0,1.0))
#     camera.track_user_inputs(ggui,movement_speed=0.03,hold_key=ti.ui.LMB)
#     scene.set_camera(camera)
#     canvas.scene(scene)
#     ggui.show()
#######################################