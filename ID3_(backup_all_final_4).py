import pandas as pd
import sys
import numpy as np
import csv
sys.__stdout__=sys.stdout


def get_attr_names(df):
    return list(df.columns.values)


def get_possible_labels(examples,attr_name):     #getting possible labels of a attribute
    attr_column=examples[attr_name]
    labels=[]
    for m in attr_column:
        if m not in labels:
            labels.append(m)
    return(labels)
    #return examples[attr_name].unique()


def get_target_label_counts(target_attr,target_labels): #getting target label counts of a dataset
    counts={}
    for label in target_labels:
        counts[label]=0
    for j in target_attr.index:
            target_val=target_attr[j]
            for label in target_labels:
                if target_val==label:
                    counts[label]=counts[label]+1
    return counts
    #return dict(target_attr.value_counts())
def get_most_common(counts):
    most_common=''
    max_count=0
    for kk,vv in counts.items():
        if int(vv)>max_count:
               most_common=kk
               max_count=vv
    return most_common
    
def entropy(counts):
    total=0
    ent=0
    counts=list(counts.values())
    total=np.sum(counts)
    if total==0:
        return ent
    for val in counts:
        p=val/total
        if p!=0:
            ent-=1*p*np.log2(p)
    return ent


def gain(examples,attr_name,target_attr,target_labels):

    tg_label_count=get_target_label_counts(target_attr,target_labels)            #Getting label(+/-/others) count for dataset/subset
    ent_s=entropy(tg_label_count)                                                #Getting entropy for the dataset/subset
    total_items=len(examples)                                                      #getting total items count in attrumn(attribute)
    gain=0
    split_info=0
    labels=get_possible_labels(examples,attr_name)                                  #Getting possible labels for the attribute
    for i in range(len(labels)):
        _,target_attrVi=get_df_with_label(examples,target_attr,attr_name,labels[i]) #extract subset with label given
        tg_label_count=get_target_label_counts(target_attrVi,target_labels)         #Getting label(+/-/others) count for dataset/subset                                                                    #Running iteration over splitted labels

        ent=entropy(tg_label_count)                                                 #Getting entropy of each split
        no_of_items=len(target_attrVi)                                                 #weighted sum of entropies of splits
        ratio=no_of_items/total_items
        gain=gain+ent*(ratio)

        
        if ratio!=0:
            split_info-=(ratio*np.log2(ratio))                           #Getting weighted sum of split info
            
    gain=ent_s-gain                                                                #Getting information gain               
    if split_info!=0:
        gain_ratio=gain/split_info                                          
    #print('Log ....',attr_name,ent_s,gain)
    return split_info,gain

def get_df_with_label(examples,target_attr,attr_name,label):                        
    attr_attr=examples.loc[:,attr_name]                                  
    ret=[]
    for index in attr_attr.index:
        if attr_attr[index]!=label:
            ret.append(index)
    examplesVi=examples.drop(ret,axis=0)
    target_attrVi=target_attr.drop(ret)
    return examplesVi,target_attrVi

def get_thrsholds(df,attr_name,target_attr):
        df[target_name]=target_attr
        df_1=df.sort_values(attr_name,axis=0)               #Sorting value
        target_attr_1=df_1[target_name]
        prev_value=target_attr_1[target_attr_1.index.values[0]] #saving first value as previous value
        thrs=[]                                                 #initializing variables
        thrs_val=[]
        summ=[]
        i=0
        #print(df_1)
        for index in (target_attr_1.index):
            thrs.append(index)                                        #saving values until any change in labels
            if target_attr_1.loc[index]!=prev_value and len(thrs)>=2: #Comparing value with previous value
                summ.append(thrs[:-1])                                #Saving the first group
                temp=thrs[-1]
                thrs=[]                                                 
                thrs.append(temp)
                val=df.loc[index,attr_name]                           #getting value of the threshold point 
                if val not in(thrs_val):                              #checking duplicate
                    thrs_val.append(val)                                
                prev_value=target_attr_1.loc[index]                   #
        summ.append(thrs)
        return(summ,thrs_val)                                       #returning threshold values and index
                
        #df get_splitted_gain


def best_split(df,attr_name,target_attr,thrs_val):
        df_left=df
        df_right=df
        wg_ent_sum=0
        total_items=(len(df))
        max_gain=0
        best_thr=0
        tg_label_count=get_target_label_counts(target_attr,target_labels)   #getting target labels counts
        ent_s=entropy(tg_label_count)
        print(len(thrs_val))
        if len(thrs_val)>200:
            thrs_val[0]=np.mean(thrs_val)
            thrs_val=thrs_val[:1]
        for val in thrs_val:                                                ####iteration over each threshold value
            df_left=df
            df_right=df
            #val=thrs_val[0]
            #for index,row in df.iterrows():                                 ####spliting for a threshold value
            filt_left=df_left[attr_name]>=val
            filt_right=df_right[attr_name]<val
            df_left=df_left[filt_left]
            df_right=df_right[filt_right]
            df_list=[df_left,df_right]
            wg_ent_sum=0
            for sub_df in df_list:                                          #ierating over splited df
                target_attrVi=sub_df[target_name]
                tg_label_count=get_target_label_counts(target_attrVi,target_labels)
                ent=entropy(tg_label_count)
                no_items=len(target_attrVi)
                wg_ent_sum+=(no_items/total_items)*ent                      ####Getting weighted sum for splited data
            #print(ent_s,wg_ent_sum)
            gain=ent_s-wg_ent_sum                                           ####Getting gain
            if gain>max_gain:
                max_gain=gain
                best_thr=val                                                ####Getting thresh value with max gain
            #print(gain,val,best_thr)
        return best_thr
        
def relabel_attr(df,attr_name,best_thr):                                    
    new_df=[]
    for index,row in df.iterrows():
        #print(df.loc[index,attr_name])
        if (df.loc[index,attr_name]>best_thr):                          #Labeling values greater than threshold with descriptive string
                new_df.append(attr_name+'>='+str(best_thr))
        if (df.loc[index,attr_name]<=best_thr):                         #Labeling values less than threshold with descriptive string
                new_df.append(attr_name+'<'+str(best_thr))
    return new_df

def discretize(df,attr_names,target_attr):
    new_dict={}
    max_label=5
    for attr_name in attr_names:
        lb=get_possible_labels(df,attr_name)
        if(len(lb)>max_label) and df.dtypes[attr_name] !='object':
            print(attr_name)
            summ,thrs_val=get_thrsholds(df,attr_name,target_attr)
            best_thr=best_split(df,attr_name,target_attr,thrs_val)
            new_attr=relabel_attr(df,attr_name,best_thr)
            new_dict[attr_name]=new_attr
        else:
            labels=get_possible_labels(df,attr_name)
            lb_num=len(labels)
            if lb_num<=max_label:
                new_dict[attr_name]=list(df[attr_name])
    #new_dict[target_name]=list(df.loc[:,target_name])
    new_df=pd.DataFrame(new_dict)
    return(new_df)
#names=['A','B','C','D','E','F','G','H','I','J']
df=pd.read_csv('abalone.csv',delimiter=',' )#,names=names)

#df=df.sample(frac=1).reset_index(drop=True)
#print(df.head(10))
#input()
df=df.fillna(df.mean())
#df=df.replace(np.NaN,0)
#df=df.drop('day',axis=1)
#df=df.drop('Ticket',axis=1)

summary=[]
res=[]
attr_names=get_attr_names(df)
target_name=attr_names[-1]
target_attr=df[target_name]
target_labels=get_possible_labels(df,target_name)

tg_label_count=get_target_label_counts(target_attr,target_labels)
most_common=get_most_common(tg_label_count)

df=df.drop(target_name,axis=1)
attr_names=attr_names[:-1]
df=discretize(df,attr_names,target_attr)
attr_names=get_attr_names(df)
print(attr_names,target_labels)
print(df.head(10))
input('Press Enter to continue:')
split=int(0.8*len(df))
#split=0
test_df=df.loc[split:,:]
test_target=target_attr.loc[split:]
test_df=test_df.reset_index(drop=True)
test_target=test_target.reset_index(drop=True)
#split=len(df)
df=df.loc[:split,:]
target_attr=target_attr.loc[:split]

#for attr_name in attr_names:
#    split_info,gain_value= gain(df,attr_name,target_attr,target_labels)
#    print(attr_name,split_info)
#    input()
def ID3(examples,target_attr,attr_names):
    global summary,res
    tg_label_count=get_target_label_counts(target_attr,target_labels)              ###Getting target label counts
    total=len(target_attr)
    for label,count in tg_label_count.items():                                     ###Check purity=1 for all target label
        if count==total:
           return label
    most_common=get_most_common(tg_label_count)
    if len(attr_names)==0:                                                          ###Check for empty dataset                  
            return most_common
    max_gain=-1                                                                     ###Getting attribute with highest gain
    attr_name=''
    for i in range(len(attr_names)):                                                 
        _,gain_value= gain(examples,attr_names[i],target_attr,target_labels)
        if gain_value>max_gain:
            max_gain=gain_value
            attr_name=attr_names[i]
    print('attr_name',attr_name)
    labels=get_possible_labels(examples,attr_name)                                      ###Getting possible labels for bes attribute
    print('labels',labels)
    for label in labels:
        examplesVi,target_attrVi=get_df_with_label(examples,target_attr,attr_name,label)###Getting new subset of dataset
        examplesVi=examplesVi.drop(attr_name,axis=1)                            
        res.append(attr_name)                                                           ###returning attr_name,label pair
        res.append(label)
        attr_names=get_attr_names(examplesVi)
        if len(attr_names)==0:# len(target_attrVi)>=2:                                    ###Checking for empty datasets                                                                                         #Getting most common label
            return most_common
        else:
            ret=ID3(examplesVi,target_attrVi,attr_names)                                ###Calling ID3
            if ret !=None:
                res.append(target_name)
                res.append(ret)
                summary.append(res)
                res=[]
    return 


#print('ret',summary)

def build_tree(summary):
    root=summary[0][0]
    for k in range(len(summary)):
        if summary[k][0]==root:
            temp=summary[k]
        if summary[k][0]!=root:
            #print(temp)
            for t in range(len(temp)):
                if temp[t]==summary[k][0]:
                    temp=temp[:t]
                    break
            summary[k]=temp+summary[k]
        print(summary[k])
fp={}
tn={}
tp={}
precision={}
recall={}
for label in target_labels:
    fp[label]=0
    tn[label]=0
    tp[label]=0
    recall[label]=0
    precision[label]=0
def predict(test_df,test_target,summary):
    acc=0
    for kk in range(len(test_df)):
         test=test_df.loc[kk,:]
         #print('..............',kk)
         for i in range(len(summary)):
            m=len(summary[i])                   #Taking a rule
            for j in range(int(m/2)):
                attr=summary[i][2*j]            #Taking a attr
                lb=summary[i][2*j+1]            #Taking corresponding label
                #print('attr',attr,'lb',lb)
                if attr!=target_name:           #last node is not reached
                   if test[attr] != lb:         #Checking test data having same label 
                        break                   #if not break and go for next rule
            if attr == target_name:             #if test data satisfy have all the attr>> label in rule
                pred_label=lb                          #setting result according to rule
                true_label=test_target[kk]
                #print(kk,'rule',i,test_target[kk],'res',res)
                if true_label==pred_label:        #Checking results validity from true label
                    acc=acc+1
                    tp[pred_label]+=1
                #break
                if true_label!=pred_label:
                     tn[pred_label]+=1
                     fp[true_label]+=1
                break                           
            if attr==target_name:
                break
    accuracy=acc/len(test_df)*100
    
    for label in target_labels:
            try:recall[label]=tp[label]/(tp[label]+tn[label])
            except:0
            try:precision[label]=tp[label]/(tp[label]+fp[label])
            except:0
            print(label,'precision',precision[label],'recall',recall[label])
    
    print('Summary : ','correct:',acc, 'total :',len(test_df),'accuracy :',acc/len(test_df)*100,'%')
    return accuracy
ID3(df,target_attr,attr_names)
build_tree(summary)
accuracy=predict(test_df,test_target,summary)
print(len(summary))
input('Press Enter for reduced error pruning')
accuracy=0
max_i=0
max_acc=0
max_summary=''

for i in range(len(summary)):
        temp_rule=summary[i]
        if (len(summary[i])>4):
            temp=summary[i][-2:]
            summary[i]=summary[i][:-4]
            summary[i]=summary[i]+temp
        print(summary[i])
        if accuracy>=max_acc:
            max_acc=accuracy
            max_summary=summary
            max_i=i
        accuracy=predict(test_df,test_target,summary)
        print(accuracy)
        summary[i]=temp_rule
print('summary',max_summary)
print('i ',max_i)
print('max_accuracy',max_acc)
#i  8
#max_accuracy 66.76300578034682
