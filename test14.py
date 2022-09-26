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

    for n in range(0,12): #计算弹簧力。
        Sping_Type=ti.Vector([Get_X(n),Get_Y(n)]) #struct，shear，bend，各四个弹簧，总共十二个弹簧。
        Coord_Neigh=coord+Sping_Type #弹簧的另一端点坐标
        if (Coord_Neigh.x>=0) and (Coord_Neigh.x<=ClothResX) and (Coord_Neigh.y>=0) and (Coord_Neigh.y<=ClothResX): #保证弹簧在布料范围内
            Sping_Vector=Sping_Type*ti.Vector([ClothWid/ClothResX,ClothHgt/ClothResX])
            Rest_Length=ti.sqrt(Sping_Vector.x**2+Sping_Vector.y**2) #计算弹簧原始长度
            p2=pos[Coord_Neigh]
            deltaP=p1-p2
            dist=ti.sqrt(deltaP.x**2+deltaP.y**2+deltaP.z**2) #计算弹簧现在的长度
            Sping_Force=ti.Vector([0.0,0.0,0.0])
            if (n<4):
                Sping_Force=(KStruct[0]*(dist-Rest_Length)*((deltaP)/dist))
            elif (n>=4 and n<8):
                Sping_Force=(KShear[0]*(dist-Rest_Length)*((deltaP)/dist))
            else:
                Sping_Force=(KBend[0]*(dist-Rest_Length)*((deltaP)/dist)) #胡克定律计算弹簧力
            F[coord]+=Sping_Force
    
@ti.func
def collision(coord): #将y=0的平面定为桌面，检查是否碰撞
    if(pos[coord].y<0):
        pos[coord].y=0 

@ti.kernel
def Reset_Cloth(): #初始化布料模型，布料正中心点为[0.0,0.0,0.0]
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
        if (i == 0 and j == 0 ) or (i == 35 and j == 0 ): #抓取布料两个顶点，改变它们的坐标来折叠布料，坐标点为（-2.0，0.0,-2.0)和(2.0,0.0,-2.0)
            z=(ClothWid/step)*t
            y=a*z**2+b
            x=0.0
            if pos[coord].x <0.0:
                x=-ClothHgt/2.0
            else:
                x=ClothHgt/2.0
            pos[coord]=[x,y,z] #顶点移动路径为f(z)=-0.3z**2+1.2
        else:                   #除布料顶点外其他点更新加速度、速度和位置。
            acc[coord]=F[coord]/mass[0]
            vel[coord]=acc[coord]*dt
            pos[coord]+=vel[coord]*dt
        collision(coord) #将y=0的平面定为桌面，检查是否碰撞

@ti.kernel
def Compute_Loss():
    current=pos[18,0]
    target=ti.Vector([0.0,0.0,2.0])
    Loss_Vector=current-target
    loss_n[None]=0.5*ti.sqrt(Loss_Vector.x**2+Loss_Vector.y**2+Loss_Vector.z**2) #将两个抓取点所在的边的正中间的点定为误差计算对象，布料初始化时，位置为[0.0,0.0,-2.0]对折后目标位置为[0.0,0.0,2.0]

@ti.kernel
def Vec_Clear(): #初始化所有向量的grad
    for i,j in pos:
        pos.grad[i,j]=ti.Vector([0.0,0.0,0.0])
        vel.grad[i,j]=ti.Vector([0.0,0.0,0.0])
        F.grad[i,j]=ti.Vector([0.0,0.0,0.0])
        acc.grad[i,j]=ti.Vector([0.0,0.0,0.0]) 

@ti.kernel
def Scalar_Clear(): #初始化所有标量的grad
    KStruct.grad[0]=0.0
    KShear.grad[0]=0.0
    KBend.grad[0]=0.0
    loss_n.grad[None]=0.0
    mass.grad[0]=0.0
    damping.grad[0]=0.0

def Grad_Clear(): #初始化所有可微变量
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


if __name__ == "__main__":
    main()