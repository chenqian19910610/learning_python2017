import random, sys
r=None
w=None
k=None
m=None
inv=None
day=1
rnd=random.Random(98765)

class orderclass:
    queue=[]
    qlen=0
    nshipped=0
    towait=0
    nshipedq=0
    towaitq=0
    qleftoverfromlastnight=False
    def _init_(self,nw):
        self.numpending=nw
        self.dayorderreceived=g.day
        self.arrivedtoq=orderclass,qleftoverfromlastnight
        orderclass.queue.append(self)
        orderclass.qlen+=1
    def simdaytime():
        orderclass.qleftoverfromlastnight=len(orderclass.queue) > 0
        numneworders=g.rnd.randit(0,g.k)
        for o in range(numneworders):
            nwidgets=g.rnd.randit(1,g.m)
            neworder=orderclass(nwidgets)
        while True:
            if orderclass.queue==[]:
                return
            o=orderclass.queue[0]
            if o.numpending<=g.inv:
                partfill=False
                orderclass.queue.remove(o)
                orderclass.qlen-=1
                sendtoday=o.numpending
                g.inv-=sendtoday
            else:
                partfill=True
                sendtoday=g.inv
                o.numpending-=g.inv
                g.inv=0
            orderclass.nshipped+=sendtoday
            waittime=g.day-o.dayorderreceived
            orderclass.towait+=sendtoday*waittime
            if o.arrivedtoq:
                orderclass.nshipedq+=sendtoday
                orderclass.towaitq+=sendtoday*waittime
            if partfill:
                return
    simdaytime=staticmethod(simdaytime)
    def simevening():
        if g.inv+g.r<=g.w:g.inv+=g.r
        else:
            g.inv=g.w
    simevening=staticmethod(simevening)
def mai():
    g.r=int(sys.argv[1])
    g.w=int(sys.argv[2])
    g.k=int(sys.argv[3])
    g.m=int(sys.argv[4])
    g.inv=int(sys.argv[5])
    ndays=int(sys.argv[6])
    for g.day in range(ndays):
        orderclass.simdaytime()
        orderclass.simevening()
    print(orderclass.towait/float(orderclass.nshipped))