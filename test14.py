import taichi as ti
import matplotlib.pyplot as plt
ti.init(arch=ti.cuda,device_memory_fraction=0.3)

ClothWid=4.0
ClothHgt=4.0
ClothResX=35
step=1024

scalar = lambda: ti.field(dtype=ti.f32)
vec = lambda: ti.Vector.field(3, dtype=ti.f32)

pos=vec()
vel=vec()
F=vec()
acc=vec()

KStruct=scalar()
KShear=scalar()
KBend=scalar()
loss_n=scalar()
mass=scalar()
damping=scalar()

ti.root.dense(ti.ij,(ClothResX+1,ClothResX+1)).place(pos,vel,F,acc)
ti.root.dense(ti.i,1).place(KStruct,KShear,KBend,mass,damping)
ti.root.place(loss_n)
ti.root.lazy_grad()

gravity=ti.Vector([0.0,-9.8,0.0])
a=-0.3
b=1.2
dt=0.01
learning_rate=1.0

KStruct[0]=20.0
KShear[0]=20.0
KBend[0]=20.0

mass[0]=0.01
damping[0]=0.025
loss_n[None]=0.0

@ti.func
def Get_X(n) ->ti.i32:
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
def Get_Y(n)->ti.i32:
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
def Compute_Force(coord):
    p1=pos[coord]
    F[coord]=gravity*mass[0]-vel[coord]*damping[0]

    for k in range(0,12):
        Sping_Type=ti.Vector([Get_X(k),Get_Y(k)])
        Coord_Neigh=coord+Sping_Type
        if (Coord_Neigh.x>=0) and (Coord_Neigh.x<=ClothResX) and (Coord_Neigh.y>=0) and (Coord_Neigh.y<=ClothResX):
            Sping_Vector=Sping_Type*ti.Vector([ClothWid/ClothResX,ClothHgt/ClothResX])
            Rest_Length=ti.sqrt(Sping_Vector.x**2+Sping_Vector.y**2)
            p2=pos[Coord_Neigh]
            deltaP=p1-p2
            dist=ti.sqrt(deltaP.x**2+deltaP.y**2+deltaP.z**2)
            Sping_Force=ti.Vector([0.0,0.0,0.0])
            if (k<4):
                Sping_Force=(KStruct[0]*(dist-Rest_Length)*((deltaP)/dist))
            elif (k>=4 and k<8):
                Sping_Force=(KShear[0]*(dist-Rest_Length)*((deltaP)/dist))
            else:
                Sping_Force=(KBend[0]*(dist-Rest_Length)*((deltaP)/dist))
            F[coord]+=Sping_Force
    
@ti.func
def collision(coord):
    if(pos[coord].y<0):
        pos[coord].y=0

@ti.kernel
def Reset_Cloth():
    for i,j in pos:
        pos[i,j] = ti.Vector([ClothWid*(i/ClothResX)-ClothWid/2.,0.0,ClothHgt*(j/ClothResX)-ClothHgt/2.0])
        vel[i,j]=ti.Vector([0.0,0.0,0.0])
        F[i,j]=ti.Vector([0.0,0.0,0.0])
        acc[i,j]=ti.Vector([0.0,0.0,0.0])

@ti.kernel
def simulation(t:ti.i32):
    for i,j in pos:
        coord=ti.Vector([i,j])
        Compute_Force(coord)
        if (i == 0 and j == 0 ) or (i == 35 and j == 0 ):
            z=(ClothWid/step)*t
            y=a*z**2+b
            x=0.0
            if pos[coord].x <0.0:
                x=-ClothHgt/2.0
            else:
                x=ClothHgt/2.0
            pos[coord]=[x,y,z]
        else:
            acc[coord]=F[coord]/mass[0]
            vel[coord]=acc[coord]*dt
            pos[coord]+=vel[coord]*dt
        collision(coord)

@ti.kernel
def Compute_Loss():
    current=pos[18,0]
    target=ti.Vector([0.0,0.0,2.0])
    Loss_Vector=current-target
    loss_n[None]=0.5*ti.sqrt(Loss_Vector.x**2+Loss_Vector.y**2+Loss_Vector.z**2)

@ti.kernel
def Vec_Clear():
    for i,j in pos:
        pos.grad[i,j]=ti.Vector([0.0,0.0,0.0])
        vel.grad[i,j]=ti.Vector([0.0,0.0,0.0])
        F.grad[i,j]=ti.Vector([0.0,0.0,0.0])
        acc.grad[i,j]=ti.Vector([0.0,0.0,0.0])

@ti.kernel
def Scalar_Clear():
    KStruct.grad[0]=0.0
    KShear.grad[0]=0.0
    KBend.grad[0]=0.0
    loss_n.grad[None]=0.0
    mass.grad[0]=0.0
    damping.grad[0]=0.0

def Grad_Clear():
    Vec_Clear()
    Scalar_Clear()

def forward():
    for t in range(step):
        simulation(t)
    Compute_Loss()

def main():
    Reset_Cloth()
    Grad_Clear()
    forward()
    losses=[]
    for i in range(step):
        with ti.ad.Tape(loss=loss_n,clear_gradients=True):
            forward()
        print("i=",i,"loss=",loss_n[None])
        losses.append(loss_n[None])
        KStruct[0]-=learning_rate * KStruct.grad[0]
        KShear[0]-=learning_rate * KShear.grad[0]
        KBend[0]-=learning_rate * KBend.grad[0]
        Reset_Cloth()
    
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    plt.plot(losses)
    plt.title("loss_change")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.show()

main()