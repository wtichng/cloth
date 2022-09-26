import taichi as ti
import time
import matplotlib.pyplot as plt
ti.init(arch=ti.cuda,device_memory_fraction=0.3,random_seed=int(time.time()))


@ti.func
def GetNextNeighborX(n):
    ax=0
    if (n==0) or (n==4) or (n==7):
        ax=1
    if (n==2) or (n==5) or (n==6):
        ax=-1
    if (n==8):
        ax=2
    if (n==10):
        ax=-2
    return ax

@ti.func
def GetNextNeighborY(n):
    ax=0
    if (n==1) or (n==4) or (n==5):
        ax=-1
    if (n==3) or (n==6) or (n==7):
        ax=1
    if (n==9):
        ax=-2
    if (n==11):
        ax=2
    return ax

@ti.func
def Get_Length3(v):
    return ti.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)

@ti.func
def Get_Length2(v):
    return ti.sqrt(v.x*v.x+ v.y*v.y)

@ti.kernel
def reset_cloth():
    for i,j in pos:
        pos[i,j]=ti.Vector([ClothWid*(i/ClothResX)-ClothWid/2.,0.0,ClothHgt*(j/ClothResX)-ClothHgt/2.0])
        Pos_Pre[i,j]=pos[i,j]
        vel[i,j]=ti.Vector([0.0,0.0,0.0])
        F[i,j]=ti.Vector([0.0,0.0,0.0])

        if i<ClothResX-1 and j<ClothResX-1:
            Tri_Id=((i*(ClothResX-1))+j)*2
            indices[Tri_Id*3+2]=i*ClothResX+j
            indices[Tri_Id*3+1]=(i+1)*ClothResX+j
            indices[Tri_Id*3+0]=i*ClothResX+(j+1)

            Tri_Id+=1
            indices[Tri_Id*3+2]=(i+1)*ClothResX+j+1
            indices[Tri_Id*3+1]=i*ClothResX+(j+1)
            indices[Tri_Id*3+0]=(i+1)*ClothResX+j

@ti.func
def Compute_Force(coord):
    p1=pos[coord]
    v1=vel[coord]
    F[coord]=gravity*mass+vel[coord]*damping
    
    for k in range(0,12): 
        
        Coord_Offcet=ti.Vector([GetNextNeighborX(k), GetNextNeighborY(k)])
        Coord_Neigh=coord+Coord_Offcet

        if (Coord_Neigh.x>=0) and (Coord_Neigh.x<=ClothResX) and (Coord_Neigh.y>=0) and (Coord_Neigh.y<=ClothResX):
            Rest_Length=Get_Length2(Coord_Offcet*ti.Vector([ClothWid/ClothResX,ClothHgt/ClothResX]))
            
            p2=pos[Coord_Neigh]
            v2=(p2-Pos_Pre[Coord_Neigh])/deltaT[0]

            deltaP=p1-p2
            deltaV=v1-v2
            dist=deltaP.norm()
            LeftTerm=0.0
            if (k<4):
                LeftTerm=-KsStruct[None]*(dist-Rest_Length)
            else:
                if(k<8):
                    LeftTerm=-KsShear[None]*(dist-Rest_Length)
                else:
                    LeftTerm=-KsBend[None]*(dist-Rest_Length)
            RightTerm=0.0
            if (k<4):
                RightTerm=KdStruct[None]*(deltaV.dot(deltaP)/dist)
            else:
                if(k<8):
                    RightTerm=KdShear[None]*(deltaV.dot(deltaP)/dist)
                else:
                    RightTerm=KdBend[None]*(deltaV.dot(deltaP)/dist)
            ti.atomic_add(F[coord], deltaP.normalized()*(LeftTerm+RightTerm))

@ti.func
def collision(coord):
    if(pos[coord].y<0):
        pos[coord].y=0

@ti.kernel
def Integrator_Verlet():
    for i, j in pos:
        coord=ti.Vector([i, j])
        Compute_Force(coord)
        index=j*(ClothResX+1)+i
        if(index==0)or(index==ClothResX):
            if T[None]<=N[0]:
                z=(ClothWid/N[0])*T[None] 
                y=a[None]*z**2+b[None]
                x=0.0
                if pos[coord][0]<0:
                    x=-ClothHgt/2.0
                else:
                    x=ClothHgt/2.0
                pos[coord]=(x,y,z)
            else:
                x=0.0
                if pos[coord][0]<0:
                    x=-ClothHgt/2.
                else:
                    x=ClothHgt/2.0
                pos[coord]=(x,0,2)
        else:
            acc=F[coord]/mass
            tmp=pos[coord]
            pos[coord]=pos[coord]*2.0-Pos_Pre[coord]+acc*deltaT[0]*deltaT[0]
            vel[coord]=(pos[coord]-Pos_Pre[coord])/deltaT[0]
            Pos_Pre[coord]=tmp
        collision(coord)

@ti.kernel
def update_verts():
    for i,j in ti.ndrange(ClothResX, ClothResX):
        vertices[i*ClothResX+j]=pos[i,j]

def simulation():
    for t_j in range(400):
        T[None]+=1
        Integrator_Verlet()
        update_verts()
        loss()

@ti.kernel
def clear():
    for i,j in pos:
        Pos_Pre.grad[i,j]=ti.Vector([0.0,0.0,0.0])
        pos.grad[i,j]=ti.Vector([0.0,0.0,0.0])
        vel.grad[i,j]=ti.Vector([0.0,0.0,0.0])
        F.grad[i,j]=ti.Vector([0.0,0.0,0.0])
    KsStruct.grad[None]=0.0
    KdStruct.grad[None]=0.0
    KsShear.grad[None]=0.0
    KdShear.grad[None]=0.0
    KsBend.grad[None]=0.0
    KdBend.grad[None]=0.0
    loss_n.grad[None]=0.0


@ti.kernel
def loss():
    target=pos[18,35]
    current=pos[18,0]
    loss_n[None]=0.5*ti.sqrt((current.x-target.x)**2+(current.y-target.y)**2+(current.z-target.z)**2)

ImgSize=512

ClothWid=4.0
ClothHgt=4.0
ClothResX=35

N_Triangles=ClothResX*ClothResX*2
indices=ti.field(dtype=ti.i32,shape=N_Triangles*3)
vertices=ti.Vector.field(3,dtype=ti.f32,shape=(ClothResX+1)*(ClothResX+1))

scalar = lambda: ti.field(dtype=ti.f32)
vec = lambda: ti.Vector.field(3, dtype=ti.f32)

Pos_Pre=vec()
pos=vec()
vel=vec()
F=vec()

KsStruct=scalar()
KdStruct=scalar()
KsShear=scalar()
KdShear=scalar()
KsBend=scalar()
KdBend=scalar()
loss_n=scalar()

ti.root.dense(ti.ij,(ClothResX+1,ClothResX+1)).place(Pos_Pre,pos,vel,F)
ti.root.place(KsStruct,KdStruct,KsShear,KdShear,KsBend,KdBend,loss_n)
ti.root.lazy_grad()

gravity=ti.Vector([0.0,-0.05,0.0]) #-0.05
a=ti.field(dtype=ti.f32,shape=())
a[None]=-0.3
b=ti.field(dtype=ti.f32,shape=())
b[None]=1.2
N=ti.field(dtype=ti.i32,shape=1)
N[0]=400
deltaT=ti.field(dtype=ti.f32,shape=1)
deltaT[0]=0.05
T=ti.field(dtype=ti.i32,shape=())
T[None]=0

learning_rate = 0.00000005

mass=1.0 #1.0
damping=-0.0125 #-0.0125
step=1024

KsStruct[None]=50.0 #50
KdStruct[None]=-0.25 #-0.25
KsShear[None]=50.0
KdShear[None]=-0.25
KsBend[None]=50.0
KdBend[None]=-0.25

ggui=ti.ui.Window('Cloth',(ImgSize,ImgSize),vsync=True)
canvas=ggui.get_canvas()
scene=ti.ui.Scene()
camera=ti.ui.Camera()
canvas.set_background_color((1, 1, 1))
camera.position(5.0,3.0,5.0)
camera.lookat(0.0,0.0,0.0)
reset_cloth()
clear()
losses=[]

def main():
    for t_i in range(10):
        reset_cloth()
        clear()
        with ti.Tape(loss_n):
            simulation()
        print("===========================")
        print("i=",t_i)
        print("loss=",loss_n[None])
        losses.append(loss_n[None])
        KsStruct[None] -= learning_rate * KsStruct.grad[None]
        KsShear[None] -= learning_rate * KsShear.grad[None]
        KsBend[None] -= learning_rate * KsBend[None]
        KdStruct[None] -= learning_rate * KdStruct.grad[None]
        KdShear[None] -= learning_rate * KdShear.grad[None]
        KdBend[None] -= learning_rate * KdBend.grad[None]

main()

print("KsStruct",KsStruct[None],KsStruct.grad[None])
print("KsShear",KsShear[None],KsShear.grad[None])
print("KsBend",KsBend[None],KsBend.grad[None])
print("KdStruct",KdStruct[None],KdStruct.grad[None])
print("KdShear",KdShear[None],KdShear.grad[None])
print("KdBend",KdBend[None],KdBend.grad[None])

reset_cloth()
clear()

fig = plt.gcf()
fig.set_size_inches(16, 9)
plt.plot(losses)
plt.title("123")
plt.xlabel("step")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()


while ggui.running:
    Integrator_Verlet()
    update_verts()
    scene.mesh(vertices,indices=indices,color=(1.,1.,1.))
    scene.point_light(pos=(10.0,10.0,0.0),color=(1.0,1.0,1.0))
    camera.track_user_inputs(ggui,movement_speed=0.03,hold_key=ti.ui.LMB)
    scene.set_camera(camera)
    canvas.scene(scene)
    ggui.show()