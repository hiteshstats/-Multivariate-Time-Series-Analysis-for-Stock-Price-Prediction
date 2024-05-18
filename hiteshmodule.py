def EDA(A):
    import seaborn as sb
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as nm
    import seaborn as sb
    import matplotlib.pyplot as plt
    from warnings import filterwarnings
    filterwarnings("ignore")
    for i in A.columns:
        if(A[i].dtypes=="float"):
            sb.distplot(A[i])
            plt.show()
        else:
            sb.countplot(A[i])
            plt.show()
            
def MDT(A):
    import pandas as pd
    Q=pd.DataFrame(A.isna().sum(),columns=["missing"])
    for i in Q[Q.missing>0].index:
        if(A[i].dtypes=="object"):
            x=A[i].mode()[0]
            A[i]=A[i].fillna(x)
        else:
            x=A[i].mean()
            A[i]=A[i].fillna(x)
    print("missing data is replaced with mean value of column for continuous data and with column mode for categorical data")
    a=pd.DataFrame(A.isna().sum(),columns=["missing"])
    return a
         
def preprocessing(A):
    import pandas as pd
    c=[]
    d=[]
 
    for i in A.columns:
        if (A[i].dtypes=="object"):
            d.append(i)
        else:
            c.append(i)
        
    x1=pd.get_dummies(A[d])
    from sklearn.preprocessing import MinMaxScaler
    mm=MinMaxScaler()
    x2=pd.DataFrame(mm.fit_transform(A[c]),columns=c)
    x=x2.join(x1)
    print("continuous data is standardized using minmax scalar and categorical data is one hot encoded")
    return x

def PreMinMax(A):
    import pandas as pd
    c=[]
    d=[]
 
    for i in A.columns:
        if (A[i].dtypes=="object"):
            d.append(i)
        else:
            c.append(i)
        
    x1=pd.get_dummies(A[d])
    from sklearn.preprocessing import MinMaxScaler
    mm=MinMaxScaler()
    x2=pd.DataFrame(mm.fit_transform(A[c]),columns=c)
    x=x2.join(x1)
    print("continuous data is standardized using MinMaxScaler and categorical data is one hot encoded")
    return x

def PreASM(A):
    a1=A.shape[0]
    a2=A.shape[1]
    B=[]
    for i in range(0,a1,1):
        a=[str(A.values[i,j]) for j in range(0,a2,1)]
        B.append(a)
    return B

def ANOVA(A,c,d):
    import pandas as pd
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    rel=c+"~"+d
    model=ols(rel,A).fit()
    av=anova_lm(model)
    Q=pd.DataFrame(av)
    a=Q["PR(>F)"][d]
    b=round(a,5)
    if(b>0.05):
        print(d,"is not good predictor for",c)
    else:
        print(d, "is good predictor for",c)
    return print("with p-value",b)

def aov(A,c,d):
    import pandas as pd
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    rel=c+"~"+d
    model=ols(rel,A).fit()
    av=anova_lm(model)
    Q=pd.DataFrame(av)
    a=Q["PR(>F)"][d]
    b=round(a,5)
    return b
  

def chisquare(A,x,y):
    import pandas as pd
    from scipy.stats import chi2_contingency
    ct=pd.crosstab(A[x],A[y])
    a,b,c,d=chi2_contingency(ct)
    if(b>0.05):
        print(x,"is not dependent on",y)

def chi(A,x,y):
    import pandas as pd
    from scipy.stats import chi2_contingency
    ct=pd.crosstab(A[x],A[y])
    a,b,c,d=chi2_contingency(ct)
    return b

def mfit_regression(model,xtr,xts,ytr,yts):
    m=model.fit(xtr,ytr)
    trp=m.predict(xtr)
    tsp=m.predict(xts)
    from sklearn.metrics import mean_absolute_error
    tre=mean_absolute_error(trp,ytr)
    tse=mean_absolute_error(tsp,yts)
    print("Training error--",tre)
    print("Testing error--",tse)
    
def mfit_classifier(model,xtr,xts,ytr,yts):
    m=model.fit(xtr,ytr)
    trp=m.predict(xtr)
    tsp=m.predict(xts)
    from sklearn.metrics import accuracy_score,recall_score
    tre=accuracy_score(trp,ytr)
    tse=accuracy_score(tsp,yts)
    recall=recall_score(tsp,yts)
    print("Training accuracy--",tre)
    print("Testing accuracy--",tse)
    print("Recall--",recall)
    

def removeTags(A):
    import re
    a=re.compile('<.*?>')
    return a.sub(r'',A)

def removeURL(A):
    import re
    a=re.compile(r'https?://\S+|www\.\S+')
    return a.sub(r"",A)

def removePunc(A):
    import string
    exclude=string.punctuation
    a=str.maketrans('','',exclude)
    return A.translate(a)

def removeSW(A): 
    new=[]
    from nltk.corpus import stopwords
    a=stopwords.words("english")
    for i in A.split():
        if i in a:
            continue
        else:
            new.append(i)
    x=new[:]
    new.clear()
    return " ".join(x)

def removeUnwanted(A):
    from re import sub
    a=sub("[!,;:?/)(_%\-''*&^$#@]","",A)
    return a

def TextPreprocessing(A):
    from hiteshmodule import removeTags
    from hiteshmodule import removeURL
    from hiteshmodule import removeUnwanted
    from hiteshmodule import removeSW
    A=A.str.lower() 
    A=A.apply(removeTags)
    A=A.apply(removeURL)
    A=A.apply(removeUnwanted)
    A=A.apply(removeSW)
    print("All the basic preprocessing that are required such as lowercasing, removing tags, removing URL, reomving punctuations,\nremoving english stop words are performed.")
    return A

def TextPrePredict(A):
    from hiteshmodule import removeTags
    from hiteshmodule import removeURL
    from hiteshmodule import removeUnwanted
    from hiteshmodule import removeSW
    A=A.lower() 
    A=removeTags(A)
    A=removeURL(A)
    A=removeUnwanted(A)
    A=removeSW(A)
    return A

def SentimentModel(xtest):
    from hiteshmodule import TextPreprocessing
    x=TextPreprocessing(xtest)
    from sklearn.feature_extraction.text import TfidfVectorizer
    xp=tfidf.transform(x)
    b=le.inverse_transform(model.predict(xp))
    p=[]
    n=[]
    for i in b:
        if(i=="positive"):
            p.append(i)
        else:
            n.append(i)
    print("------Number of Positive reviews ------> ",len(p))
    print("------Number of Negative reviews ------> ",len(n))
    p.extend(n)
    new=pd.DataFrame(p,columns=["Predicted"])
    return new.Predicted.value_counts().plot(kind="bar")

def MDTnlp(A):
    import pandas as pd
    Q=pd.DataFrame(A.isna().sum(),columns=["missing"])
    for i in Q[Q.missing>0].index:
        if(A[i].dtypes=="object"):
            A[i]=A[i].fillna('')
        else:
            x=A[i].mean()
            A[i]=A[i].fillna(x)
    print("missing data is replaced with mean value of column for continuous data and with empty string for NaN data")
    a=pd.DataFrame(A.isna().sum(),columns=["missing"])
    return a
