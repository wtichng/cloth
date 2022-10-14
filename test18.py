import taichi as ti
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu,device_memory_fraction=0.3)

ImgSize=512
ClothWid=4.0
ClothHgt=4.0
ClothResX=35
# ClothResX=35
step=1024
step_z=ti.field(dtype=ti.f32,shape=())
step_z[None]=400.0

scalar = lambda: ti.field(dtype=ti.f32)
vec = lambda: ti.Vector.field(3, dtype=ti.f32)

pos=vec()
vel=vec()
F=vec()
acc=vec()
Spring_Date=vec()

Spring_K=scalar()
loss_n=scalar()
mass=scalar()
airdamping=scalar()

ti.root.dense(ti.ijk,(step,ClothResX+1,ClothResX+1)).place(pos,vel,F,acc)
ti.root.dense(ti.ijk,(ClothResX+1,ClothResX+1,12)).place(Spring_Date)
ti.root.dense(ti.ij,(3,2)).place(Spring_K)
ti.root.place(mass,airdamping,loss_n)
ti.root.lazy_grad()

gravity=ti.Vector([0.0,-0.0098,0.0])
# gravity=ti.Vector([0.0,-0.0098,0.0])
a=-0.5
b=2.0
dt=0.05
learning_rate=1.0

mass[None]=10.0
# mass[None]=10
airdamping[None]=0.0125
# airdamping[None]=0.0125
loss_n[None]=0.0

z=ti.field(dtype=ti.f32,shape=())
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
def Spring_Date_Init():
    for i in range(3):
        #0:struct 1:shear 2:bend
        Spring_K[i,0]=600.0
        Spring_K[i,1]=2.0

    for i,j,k in Spring_Date:
        #i,j:dingdian coord;
        #k:12 spring type
        Spring_Type = ti.Vector([Get_X(k),Get_Y(k)])
        Coord_Neigh = Spring_Type + ti.Vector([i,j])
        Rest_Length_Struct=(ClothWid/ClothResX)
        Rest_Length_Shear=ti.Vector([ClothWid/ClothResX,ClothHgt/ClothResX]).norm()
        Rest_length_Bend=(ClothWid/ClothResX)*2
        if (Coord_Neigh.x>=0) and (Coord_Neigh.x<=ClothResX) and (Coord_Neigh.y>=0) and (Coord_Neigh.y<=ClothResX): 
            if (k<4):
                #0,1,2,3 struct
                Spring_Date[i,j,k]=ti.Vector([Rest_Length_Struct,Coord_Neigh.x,Coord_Neigh.y])
            elif (k>=4 and k<8):
                #4,5,6,7 shear
                Spring_Date[i,j,k]=ti.Vector([Rest_Length_Shear,Coord_Neigh.x,Coord_Neigh.y])
            else:
                #8,9,10,11 bend
                Spring_Date[i,j,k]=ti.Vector([Rest_length_Bend,Coord_Neigh.x,Coord_Neigh.y])
        else:
            Spring_Date[i,j,k]=ti.Vector([0.0,0.0,0.0])

@ti.func
def Compute_Force(t:ti.i32,i:ti.i32,j:ti.i32):
    p1=pos[t-1,i,j]
    v1=vel[t-1,i,j]
    F[t,i,j]=gravity*mass[None]-vel[t-1,i,j]*airdamping[None]
    for n in range(0,12):
        Spring_Length=Spring_Date[i,j,n][0]
        if Spring_Length != 0 :
            x=ti.cast(Spring_Date[i,j,n][1],ti.i32)
            y=ti.cast(Spring_Date[i,j,n][2],ti.i32)
            p2=pos[t-1,x,y]
            v2=(p2-pos[t-2,x,y])/dt
            dv=v1-v2
            dp=p1-p2
            dist=dp.norm()
            AX=-Spring_K[n//4,0]*(dist-Spring_Length)
            BX=-Spring_K[n//4,1]*(dv.dot(dp)/dist)
            F[t,i,j] += dp.normalized()*(AX+BX)

@ti.kernel
def Reset_Cloth(): 
    Spring_Date_Init()
    for t,i,j in pos:
        pos[t,i,j] = ti.Vector([ClothWid*(i/ClothResX)-ClothWid/2.,0.0,ClothHgt*(j/ClothResX)-ClothHgt/2.0])
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
    for i,j in ti.ndrange(ClothResX+1,ClothResX+1):
        Compute_Force(t,i,j)
        if ((i == 0 and j == 0 ) or (i == ClothResX and j == 0 )): 
            if z[None] <= ClothHgt/2.0:
                z[None] += (ClothWid/step_z[None])
            else:
                z[None] = 2.0 
            y=a*z[None]**2+b
            x=0.0
            if i==0:
                x=-ClothHgt/2.0
            else:
                x=ClothHgt/2.0
            pos[t,i,j]=ti.Vector([x,y,z[None]])
            acc[t,i,j]=F[t,i,j]/mass[None]
            vel[t,i,j]=(pos[t,i,j]-pos[t-2,i,j])/dt
        else:
            acc[t,i,j]=F[t,i,j]/mass[None]
            ax=pos[t-1,i,j]*2-pos[t-2,i,j]+acc[t,i,j]*dt*dt
            if ax.y < 0:
                ax.y=0
            pos[t,i,j]=ax
            vel[t,i,j]=(pos[t,i,j]-pos[t-2,i,j])/dt


@ti.kernel
def Compute_Loss(t:ti.i32):
    current=pos[t,18,0]
    target=pos[t,18,35]
    Loss_Vector=current-target
    loss_n[None]=Loss_Vector.norm()

@ti.kernel
def Vec_Clear(): 
    for t,i,j in pos:
        pos.grad[t,i,j]=ti.Vector([0.0,0.0,0.0])
        vel.grad[t,i,j]=ti.Vector([0.0,0.0,0.0])
        F.grad[t,i,j]=ti.Vector([0.0,0.0,0.0])
        acc.grad[t,i,j]=ti.Vector([0.0,0.0,0.0]) 

@ti.kernel
def Spring_K_Clear():
    for i , j in ti.ndrange(3,2):
        Spring_K.grad[i,j]=0.0

@ti.kernel
def Scalar_Clear(): 
    loss_n.grad[None]=0.0
    mass.grad[None]=0.0
    airdamping.grad[None]=0.0

def Grad_Clear(): 
    Vec_Clear()
    Scalar_Clear()
    Spring_K_Clear()

def forward():
    for t in range(2,step):
        simulation(t)
    Compute_Loss(int(step-1))
##########################################
@ti.kernel
def update_verts(t:ti.i32):
    for i,j in ti.ndrange(ClothResX, ClothResX):
        vertices[i*ClothResX+j]=pos[t,i,j]
##########################################

def dmain():
    Reset_Cloth()
    # forward()
    Grad_Clear()
    # Reset_Cloth()
    losses=[]
    for i in range(step):
        Reset_Cloth()
        Grad_Clear()
        with ti.Tape(loss=loss_n,clear_gradients=True):
            for t in range(2,100):
                simulation(t)
            Compute_Loss(int(step-1))
        print("i=",i,"loss=",loss_n[None])
        losses.append(loss_n[None])
        for i , j in ti.ndrange(3,2):
            Spring_K[i,j]-=learning_rate * Spring_K.grad[i,j]
        print("KStruct.grad0:",Spring_K.grad[0,0])
        print("KShear.grad0:",Spring_K.grad[1,0])
        print("KBend.grad0:",Spring_K.grad[2,0])
        print("KStruct.grad1:",Spring_K.grad[0,1])
        print("KShear.grad1:",Spring_K.grad[1,1])
        print("KBend.grad1:",Spring_K.grad[2,1])
        print("loss_n.grad:",loss_n.grad[None])
        print("mass.grad:",mass.grad[None])
        print("airdamping.grad:",airdamping.grad[None])
        print("pos[t,18,0].grad:",pos.grad[1020,18,0])
        print("pos[t,18,35].grad:",pos.grad[1020,18,35])
        print("pos[t,18,0] ",pos[1020,18,0])
        print("pos[t,18,35]",pos[1020,18,35])
        print("vel[t,18.0].grad:",vel.grad[1020,18,0])
        print("F[t,18,0]:",F.grad[1020,18,0])
        print("acc.grad[t,18,0]",acc.grad[1020,18,0])

    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    plt.plot(losses)
    plt.title("loss_change")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.show()

####################################
def vmain():
    Reset_Cloth()
    ggui=ti.ui.Window('Cloth',(ImgSize,ImgSize),vsync=True)
    canvas=ggui.get_canvas()
    scene=ti.ui.Scene()
    camera=ti.ui.Camera()
    canvas.set_background_color((1, 1, 1))
    camera.position(5.0,3.0,5.0)
    camera.lookat(0.0,0.0,0.0)
    for i in range(2,100):
        simulation(i)
    while ggui.running:
        for i in range (step):
            update_verts(i)
            scene.mesh(vertices,indices=indices,color=(1.,1.,1.))
            scene.point_light(pos=(10.0,10.0,0.0),color=(1.0,1.0,1.0))
            camera.track_user_inputs(ggui,movement_speed=0.03,hold_key=ti.ui.LMB)
            scene.set_camera(camera)
            canvas.scene(scene)
            ggui.show()
######################################

if __name__ == "__main__":
    # vmain()
    dmain()