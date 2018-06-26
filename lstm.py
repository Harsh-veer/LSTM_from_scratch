import numpy as np

data=open("text0.txt","r").read()
chars=list(set(data))
vocab_size=len(chars)

features=dict()
X=[]
Y=[]

for i in chars:
    t=[]
    for j in chars:
        if i==j:
            t.append(1)
        else:
            t.append(0)
    features[i]=t

for i in data:
    X.append(features[i])


for i in X[1:]:
    Y.append(i)

X=np.array(X[0:-1])
Y=np.array(Y)
batch_size=25
beg=0
end=batch_size

def getBatch():
    global beg,end
    x_batch=[]
    y_batch=[]
    if end>len(X):
        beg=0
        end=batch_size
    for i in range(beg,end):
        x_batch.append(X[i])
        y_batch.append(Y[i])

    beg=end
    end+=batch_size

    return np.array(x_batch),np.array(y_batch)

def getkey(arr,dic):
    for k in dic.keys():
        if list(dic[k])==list(arr):
            return k

def softmax(y):
    return np.exp(y)/np.sum(np.exp(y))

def sgm(y,diff=False):
    if diff:
        return sgm(y)*sgm(y)*np.exp(-y)
    else:
        return 1/(1+np.exp(-y))

def dtanh(y):
    return (1-np.tanh(y)*np.tanh(y))

hidden_size=100
input_size=vocab_size
output_size=vocab_size
learning_rate=0.1

Wgx=np.random.randn(hidden_size,input_size)*0.01
Wgh=np.random.randn(hidden_size,hidden_size)*0.01
Wix=np.random.randn(hidden_size,input_size)*0.01
Wih=np.random.randn(hidden_size,hidden_size)*0.01
Wox=np.random.randn(hidden_size,input_size)*0.01
Woh=np.random.randn(hidden_size,hidden_size)*0.01
Wfx=np.random.randn(hidden_size,input_size)*0.01
Wfh=np.random.randn(hidden_size,hidden_size)*0.01
Why=np.random.randn(output_size,hidden_size)*0.01
bg=np.zeros([hidden_size,1])
bi=np.zeros([hidden_size,1])
bo=np.zeros([hidden_size,1])
bf=np.zeros([hidden_size,1])
by=np.zeros([output_size,1])

mWgh,mWih,mWfh,mWoh=np.zeros_like(Wgh),np.zeros_like(Wih),np.zeros_like(Wfh),np.zeros_like(Woh)
mWgx,mWix,mWfx,mWox=np.zeros_like(Wgx),np.zeros_like(Wix),np.zeros_like(Wfx),np.zeros_like(Wox)
mbg,mbi,mbf,mbo=np.zeros_like(bg),np.zeros_like(bi),np.zeros_like(bf),np.zeros_like(bo)
mWhy,mby=np.zeros_like(Why),np.zeros_like(by)

def model(x,y,pre_hs,mem,prei,preo,pref): # pre_hs and mem are from the last batch
    loss=0
    g,i,f,o={},{},{},{}
    zi,zf,zo={},{},{}
    hs,ys,ps,s={},{},{},{}
    hs[-1]=np.copy(pre_hs)
    s[-1]=np.copy(mem)
    i[-1]=np.copy(prei)
    o[-1]=np.copy(preo)
    f[-1]=np.copy(pref)
    for t in range(len(x)):
        inputs=np.array(np.matrix(x[t]).T)
        labels=np.array(np.matrix(y[t]).T)
        g[t]=np.tanh(np.dot(Wgx,inputs)+np.dot(Wgh,hs[t-1])+bg)
        zi[t]=np.dot(Wix,inputs)+np.dot(Wih,hs[t-1])+bi
        i[t]=sgm(zi[t])
        zo[t]=np.dot(Wox,inputs)+np.dot(Woh,hs[t-1])+bo
        o[t]=sgm(zo[t])
        zf[t]=np.dot(Wfx,inputs)+np.dot(Wfh,hs[t-1])+bf
        f[t]=sgm(zf[t])
        s[t]=(g[t]*i[t]) + (f[t]*s[t-1])
        hs[t]=np.tanh(s[t])*o[t]
        ys[t]=np.dot(Why,hs[t])+by
        ps[t]=softmax(ys[t])

        loss+=-np.log(ps[t][list(labels).index(1)])

    dWgh,dWih,dWfh,dWoh=np.zeros_like(Wgh),np.zeros_like(Wih),np.zeros_like(Wfh),np.zeros_like(Woh)
    dWgx,dWix,dWfx,dWox=np.zeros_like(Wgx),np.zeros_like(Wix),np.zeros_like(Wfx),np.zeros_like(Wox)
    dbg,dbi,dbf,dbo=np.zeros_like(bg),np.zeros_like(bi),np.zeros_like(bf),np.zeros_like(bo)
    dWhy,dby=np.zeros_like(Why),np.zeros_like(by)
    dhnext,dsnext=np.zeros([hidden_size,1]),np.zeros([hidden_size,1])

    for t in reversed(range(len(x))):
        inputs=np.array(np.matrix(x[t]).T)
        dy=np.copy(ps[t])
        dy[list(y[t]).index(1)]-=1

        dWhy+=np.dot(dy,hs[t].T)
        dby+=dy

        dh=np.dot(Why.T,dy)+dhnext
        dWoh+=np.dot(dh*np.tanh(s[t])*sgm(zo[t],True), hs[t-1].T)
        dWox+=np.dot(dh*np.tanh(s[t])*sgm(zo[t],True), inputs.T)
        dbo+=dh*np.tanh(s[t])*sgm(zo[t],True)

        ds=dh*o[t]*dtanh(s[t]) + dsnext
        dWgx+=np.dot(ds*i[t]*(1-g[t]*g[t]),inputs.T)
        dWgh+=np.dot(ds*i[t]*(1-g[t]*g[t]),hs[t-1].T)
        dbg+=ds*i[t]*(1-g[t]*g[t])
        dWix+=np.dot(ds*g[t]*sgm(zi[t],True),inputs.T)
        dWih+=np.dot(ds*g[t]*sgm(zi[t],True),hs[t-1].T)
        dbi+=ds*g[t]*sgm(zi[t],True)
        dWfx+=np.dot(ds*s[t-1]*sgm(zf[t],True),inputs.T)
        dWfh+=np.dot(ds*s[t-1]*sgm(zf[t],True),hs[t-1].T)
        dbf+=ds*s[t-1]*sgm(zf[t],True)

        dhpreh=dtanh(s[t])*o[t]*(np.dot(Wfh.T,sgm(zf[t],True))*s[t-1] + np.dot(Wgh.T,1-g[t]*g[t])*i[t] + np.dot(Wih.T,sgm(zi[t],True))*g[t]) + np.dot(Woh.T,sgm(zo[t],True))*np.tanh(s[t])
        dhnext=dhpreh*dh
        dhpres=dtanh(s[t])*(i[t]*(np.dot(Wgh.T,1-g[t]*g[t])*o[t-1]*dtanh(s[t-1])) + g[t]*(np.dot(Wih.T,sgm(zi[t],True))*o[t-1]*dtanh(s[t-1])) + np.dot(Wfh.T,sgm(zf[t]))*o[t-1]*dtanh(s[t-1]) + f[t])*o[t] + np.dot(Woh.T,sgm(zo[t],True))*np.tanh(s[t])*o[t-1]*dtanh(s[t-1])
        dsnext=dhpres*dh

    for dparam in [dWgh,dWih,dWfh,dWoh,dWgx,dWix,dWfx,dWox,dbg,dbi,dbf,dbo,dWhy,dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss,dWgh,dWih,dWfh,dWoh,dWgx,dWix,dWfx,dWox,dbg,dbi,dbf,dbo,dWhy,dby,hs[len(x)-1],s[len(x)-1],i[len(x)-1],o[len(x)-1],f[len(x)-1]


def train():
    pre_hs=np.zeros([hidden_size,1])
    mem=np.zeros([hidden_size,1]) # memory cell
    prei=np.zeros([hidden_size,1])
    preo=np.zeros([hidden_size,1])
    pref=np.zeros([hidden_size,1])
    best=10000
    n=0
    while True:
        x_batch,y_batch=getBatch()
        loss,dWgh,dWih,dWfh,dWoh,dWgx,dWix,dWfx,dWox,dbg,dbi,dbf,dbo,dWhy,dby,pre_hs,mem,prei,preo,pref = model(x=x_batch,y=y_batch,pre_hs=pre_hs,mem=mem,prei=prei,preo=preo,pref=pref)

        for param,dparam,memparam in zip([Wgh,Wih,Wfh,Woh,Wgx,Wix,Wfx,Wox,bg,bi,bf,bo,Why,by],[dWgh,dWih,dWfh,dWoh,dWgx,dWix,dWfx,dWox,dbg,dbi,dbf,dbo,dWhy,dby],[mWgh,mWih,mWfh,mWoh,mWgx,mWix,mWfx,mWox,mbg,mbi,mbf,mbo,mWhy,mby]):
            memparam += dparam*dparam
            param += -learning_rate * dparam / np.sqrt(memparam + 1e-8)

        if n%100==0:
            if loss<best:
                best=loss
            print ("iter: ",n," loss: ",loss," best: ",best)
        n+=1


def predict(begchar,outLen):
    xin=features[begchar]
    inputs=np.array(np.matrix(xin).T)
    tprehs=np.zeros([hidden_size,1])
    tpres=np.zeros([hidden_size,1])
    genstr=""

    for _ in range(outLen):
        g=np.tanh(np.dot(Wgx,inputs)+np.dot(Wgh,tprehs)+bg)
        i=sgm(np.dot(Wix,inputs)+np.dot(Wih,tprehs)+bi)
        o=sgm(np.dot(Wox,inputs)+np.dot(Woh,tprehs)+bo)
        f=sgm(np.dot(Wfx,inputs)+np.dot(Wfh,tprehs)+bf)
        s=(g*i) + (f*tpres)
        hs=np.tanh(s)*o
        ys=np.dot(Why,hs)+by
        ps=softmax(ys)

        pred=np.zeros([input_size,1])
        pred[list(ps).index(max(ps))]=1
        k=getkey(pred,features)
        genstr+=k

        inputs=np.copy(pred)
        tprehs=np.copy(hs)
        tpres=np.copy(s)

    print (genstr)
