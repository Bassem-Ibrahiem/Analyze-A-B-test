#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly.**
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[75]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[76]:


df=pd.read_csv('ab_data.csv')
df.head()


# b. Use the cell below to find the number of rows in the dataset.

# In[77]:


df.shape[0]


# c. The number of unique users in the dataset.

# In[78]:


df['user_id'].nunique()


# d. The proportion of users converted.

# In[79]:


df.query('converted == 1').user_id.nunique()/df.shape[0]


# e. The number of times the `new_page` and `treatment` don't match.

# In[80]:


df.query('(group == "treatment" and landing_page!= "new_page" or group != "treatment" and landing_page== "new_page")').user_id.count()


# f. Do any of the rows have missing values?

# In[81]:


df.isnull().sum().sum()


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[82]:


# Double Check all of the correct rows were removed - this should be 0
df2=df.drop(df.query('(group == "treatment" and landing_page!= "new_page" or group != "treatment" and landing_page== "new_page")').index)
df2.query('(group == "treatment" and landing_page!= "new_page" or group != "treatment" and landing_page== "new_page")').user_id.count()


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[83]:


df2.user_id.nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[84]:


df2[df2.user_id.duplicated()].user_id


# c. What is the row information for the repeat **user_id**? 

# In[85]:


df2[df2.user_id.duplicated()]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[86]:


df2=df2.drop_duplicates('user_id',keep='first')
df2.head(10)


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[87]:


df2.query('converted ==1').user_id.nunique()/df2.shape[0]


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[88]:


control_converted=df2.query('group=="control" and converted ==1').user_id.nunique()/df2.query('group=="control"').user_id.nunique()
control_converted


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[89]:


treatment_converted=df2.query('group=="treatment" and converted ==1').user_id.nunique()/df2.query('group=="treatment"').user_id.nunique()
treatment_converted


# d. What is the probability that an individual received the new page?

# In[90]:


new_page=df2.query('landing_page =="new_page"').user_id.nunique()/df2.shape[0]
new_page


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# **From what we saw previously that:**
# 
#     (a) Probability of who converted regardless of the page they receive= 11.96 %
#     (b) Probability of who converted - Given of an individual was in the control group = 12.04 %
#     (c) Probability of who converted - Given of an individual was in the treatment group = 11.88 % 
#     (d) Probability of an individual received the new page = 50 %
#     
# **That is no sufficient evidence to conclude that the new treatment page leads to more conversions.**

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# *$H_0$: $p_{new}$ - $p_{old}$ <= 0*
# 
# *$H_1$: $p_{new}$ - $p_{old}$ > 0*

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[91]:


p_new=df2.query('converted ==1').user_id.nunique()/df2.user_id.nunique()
p_new


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[92]:


p_old=df2.query('converted ==1').user_id.nunique()/df2.user_id.nunique()
p_old


# In[93]:


n_new=df2.query('landing_page =="new_page"').user_id.nunique()
n_old=df2.query('landing_page =="old_page"').user_id.nunique()
n_new,n_old


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[94]:


n_new1=df2.query('group == "treatment" and converted ==1').user_id.nunique()
n_new1


# d. What is $n_{old}$, the number of individuals in the control group?

# In[95]:


n_old1=df2.query('group == "control" and converted ==1').user_id.nunique()
n_old1


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[96]:


new_page_converted=np.random.binomial(1,p_new,n_new)
new_page_converted.mean()


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[97]:


old_page_converted=np.random.binomial(1,p_old,n_old)
old_page_converted.mean()


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[98]:


new_page_converted.mean() - old_page_converted.mean()


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[99]:


p_diffs=[]
new_converted_sample=np.random.binomial(n_new, p_new, 10000)/n_new
old_converted_sample=np.random.binomial(n_old, p_old, 10000)/n_old
p_diffs.append(new_converted_sample-old_converted_sample)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[100]:


plt.hist(p_diffs);


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[101]:


p_diffs=np.array(p_diffs)
actual_diff=treatment_converted-control_converted
(p_diffs>actual_diff).mean()


# In[102]:


null_vals = np.random.normal(0, p_diffs.std(), p_diffs.size)
plt.hist(null_vals)

plt.axvline(actual_diff, c='red')


# In[103]:


(null_vals > actual_diff).mean()


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **We computed the P-value in j, About 90 % and it is larger than type I error $\alpha$ 5%, So we fail to reject the null hypothesis $H_0$.**

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[104]:


import statsmodels.api as sm
convert_old = df2.query('landing_page=="old_page"').converted.sum()
convert_new = df2.query('landing_page=="new_page"').converted.sum()
n_old = df2.query('landing_page=="old_page"').user_id.nunique()
n_new = df2.query('landing_page=="new_page"').user_id.nunique()
convert_old,convert_new,n_old,n_new


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[105]:


z_score,p_value=sm.stats.proportions_ztest(np.array([convert_new,convert_old]),np.array([n_new,n_old]),alternative="larger")
z_score,p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **The Z_score and P_value from the previous test doesn't reject the null hypothesis, the z-score about -1.31 less than critical value 1.96, and the p-value about 90% larger than type I error $\alpha$ 5%, from both that means there is statistical significance to affirm the previous results in J that fail to reject the null hypothesis $H_0$.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **This is a logistic regression, since we want to know the odds of conversion, rather than a linear figure.**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[106]:


df2.head()


# In[107]:


df2['intercept']=1
df2['ab_page']=pd.get_dummies(df2['group'])['treatment']
df2.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[108]:


lm=sm.Logit(df2.converted,df2[['intercept','ab_page']])
log=lm.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[109]:


log.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# **The p-value associated with ab_page is about 0.19, this different from Part II.**
# 
# __Part II was based on a hypothesis that:__
# 
# *$H_0$: $p_{new}$ - $p_{old}$ <= 0*
# 
# *$H_1$: $p_{new}$ - $p_{old}$ > 0*
# 
# __But Part III was based on a hypothesis that:__
# 
# *$H_0$: $p_{new}$ = $p_{old}$*
# 
# *$H_1$: $p_{new}$ $\neq $ $p_{old}$*
# 
# **In both Part I & Part II results, We don't have any sufficient evidence to refuse A null hypothesis $H_0$
# , Which means the new page isn't better than the old page.**

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **It's a good idea to use another factor maybe influence individual converts, it will help more in model fitting, But disadvantages may be like there exist some Multicollinearity.**

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[110]:


df3=pd.read_csv('countries.csv')
df2=df2.set_index('user_id').join(df3.set_index('user_id'))
df2.head()


# In[72]:


df2[['UK','US']]=pd.get_dummies(df2.country)[['UK','US']]
df2.head()


# In[73]:


lm=sm.Logit(df2.converted,df2[['intercept','ab_page','UK','US']])
log=lm.fit()
log.summary()


# **Based on the p-values above, it also does not appear as though country has a significant impact on conversion.**

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[74]:


df2['ab_UK'] = df2['ab_page'] * df2['UK']
df2['ab_US'] = df2['ab_page'] * df2['US']
lm2 = sm.Logit(df2.converted, df2[['intercept', 'ab_page', 'UK' , 'US', 'ab_UK', 'ab_US']])
log2 = lm2.fit()
log2.summary()


# **All The above calculation regression summary gives us an interpretation with holding all other variables constant that the countries aren't impacted by the type of page based on their P-values does not provide a statistical sufficient evidence to reject the null hypothesis $H_0$, compare with a Type I error rate $\alpha$ 5%. Thus ultimately, based on any of our A/B testings. As a result, there is no reason to switch to the new page, when the old one performs just as well.**

# <a id='conclusions'></a>
# ## Finishing Up
# 
# > Congratulations!  You have reached the end of the A/B Test Results project!  You should be very proud of all you have accomplished!
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[111]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# In[ ]:




