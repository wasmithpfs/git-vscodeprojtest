#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[35]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[36]:


data = pd.read_csv("C:\\Users\\Wm Allen Smith\\Documents\\Bank\\banking.csv", header =0)
data = data.dropna()


# In[37]:



print(data.shape)
print(list(data.columns))


# In[38]:


data.head()


# In[39]:


data[ 'education'].unique()


# In[40]:


data['education']=np.where (data['education']=='basic.9y', 'Basic', data['education'])
data['education']=np.where (data['education']=='basic.6y', 'Basic', data['education'])
data['education']=np.where (data['education']=='basic.4y', 'Basic', data['education'])


# In[41]:


data[ 'education'].unique()


# In[42]:


data['y'].value_counts()


# In[43]:


sns.countplot(x='y', data=data , palette='hls')
plt.show()
plt.savefig('count_plot')


# In[44]:


count_no_sub=len(data[data['y']==0])
count_sub=len(data[data['y']==1])
pct_of_no_sub=count_no_sub/(count_sub+count_no_sub)
print("Percentage of no subscription is ",pct_of_no_sub*100)
pct_of_sub=count_sub/(count_sub+count_no_sub)
print("Percentage of  subscription is ", pct_of_sub*100)


# In[45]:


data.groupby('y').mean()


# In[46]:


data.groupby('job').mean()


# In[47]:


data.groupby('marital').mean()


# In[48]:


data.groupby('education').mean()


# In[55]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(data.job,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')


# In[58]:


table =pd.crosstab(data,data.marital, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar' , stacked= True)
plt.title('Stacked Bar Chart of Marital Stautis vs Purchase')
plt.xlabel=('Marital Status')
plt.ylabel-('Proportion of Customers')
plt.savefig('marital_vs_pur_stack')


# In[ ]:





# In[ ]:





# In[57]:


pd.crosstab(data.month, data.y).plot(kind='bar')
plt.title('Purchase Frequency of Month')     
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')  
plt.savefig('purchase_fre_month_bar')                                     
                                     


# In[ ]:





# In[59]:


pd.crosstab(data.day_of_week, data.y).plot(kind='bar')
plt.title('Purchase Frequency of Day of Week')     
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')  
plt.savefig('purchase_fre_day_bar')                                     
                                     


# In[60]:


data.age.hist()
plt.title('Histogram by age')     
plt.xlabel('age')
plt.ylabel('Frequency')  
plt.savefig('purchase_fre__age')                                     
                                     


# In[61]:


table=pd.crosstab(data.marital,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('marital_vs_pur_stack')


# In[62]:


table=pd.crosstab(data.education,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')
plt.savefig('edu_vs_pur_stack')


# In[63]:


table=pd.crosstab(data.day_of_week,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_dayofweek_bar')


# In[64]:


pd.crosstab(data.month,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_month_bar')


# In[65]:


data.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')


# In[66]:


pd.crosstab(data.poutcome,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Poutcome')
plt.xlabel('Poutcome')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_pout_bar')


# In[67]:


cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
    
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]


# In[ ]:





# In[68]:


data_final =data[to_keep]
data_final.columns.values


# In[69]:


print(data_final)


# In[70]:


print(data_final.columns.values)


# In[71]:


X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))


# In[82]:



data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear')
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)


# In[ ]:





# In[84]:


cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown', 
      'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']


# In[95]:


import statsmodels.api as sm
logit_model=sm.Logit(y,X)

#set maxiter to 200 or more to remove convergence warning
result=logit_model.fit(maxiter=200)
print(result.summary2())


# In[96]:


cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 
      'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']
logit_model=sm.Logit(y,X)
result=logit_model.fit(maxiter=200)
print(result.summary2())


# In[97]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)


# In[98]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[99]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[100]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[101]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

