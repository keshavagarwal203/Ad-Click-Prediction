# Ad-Click-Prediction
1. Business Problem
1.1 Problem Description
Introduction:
Clickthrough rate (CTR) is a ratio showing how often people who see your ad end up clicking it. Clickthrough rate (CTR) can be used to gauge how well your keywords and ads are performing.

CTR is the number of clicks that your ad receives divided by the number of times your ad is shown: clicks ÷ impressions = CTR. For example, if you had 5 clicks and 100 impressions, then your CTR would be 5%.

Each of your ads and keywords have their own CTRs that you can see listed in your account.

A high CTR is a good indication that users find your ads helpful and relevant. CTR also contributes to your keyword's expected CTR, which is a component of Ad Rank. Note that a good CTR is relative to what you're advertising and on which networks.
Credits: Google (https://support.google.com/adwords/answer/2615875?hl=en)

Search advertising has been one of the major revenue sources of the Internet industry for years. A key technology behind search advertising is to predict the click-through rate (pCTR) of ads, as the economic model behind search advertising requires pCTR values to rank ads and to price clicks. In this task, given the training instances derived from session logs of the Tencent proprietary search engine, soso.com, participants are expected to accurately predict the pCTR of ads in the testing instances.

1.2 Source/Useful Links
Source : https://www.kaggle.com/c/kddcup2012-track2
Dropbox Links : https://www.dropbox.com/sh/k84z8y9n387ptjb/AAA8O8IDFsSRhOhaLfXVZcJwa?dl=0
Blog :https://hivemall.incubator.apache.org/userguide/regression/kddcup12tr2_dataset.html

1.3 Real-world/Business Objectives and Constraints
Objective: Predict the pClick (probability of click) as accurately as possible.

Constraints: Low latency, Interpretability.

2. Machine Learning problem
2.1 Data
2.1.1 Data Overview
Filename	Available Format
training	.txt (9.9Gb)
queryid_tokensid	.txt (704Mb)
purchasedkeywordid_tokensid	.txt (26Mb)
titleid_tokensid	.txt (172Mb)
descriptionid_tokensid	.txt (268Mb)
userid_profile	.txt (284Mb)
Feature	Description
UserID	The unique id for each user
AdID	The unique id for each ad
QueryID	The unique id for each Query (it is a primary key in Query table(queryid_tokensid.txt))
Depth	The number of ads impressed in a session is known as the 'depth'.
Position	The order of an ad in the impression list is known as the ‘position’ of that ad.
Impression	The number of search sessions in which the ad (AdID) was impressed by the user (UserID) who issued the query (Query).
Click	The number of times, among the above impressions, the user (UserID) clicked the ad (AdID).
TitleId	A property of ads. This is the key of 'titleid_tokensid.txt'. [An Ad, when impressed, would be displayed as a short text known as ’title’, followed by a slightly longer text known as the ’description’, and a URL (usually shortened to save screen space) known as ’display URL’.]
DescId	A property of ads. This is the key of 'descriptionid_tokensid.txt'. [An Ad, when impressed, would be displayed as a short text known as ’title’, followed by a slightly longer text known as the ’description’, and a URL (usually shortened to save screen space) known as ’display URL’.]
AdURL	The URL is shown together with the title and description of an ad. It is usually the shortened landing page URL of the ad, but not always. In the data file, this URL is hashed for anonymity.
KeyId	A property of ads. This is the key of 'purchasedkeyword_tokensid.txt'.
AdvId	a property of the ad. Some advertisers consistently optimize their ads, so the title and description of their ads are more attractive than those of others’ ads.
There are five additional data files, as mentioned in the above section:

queryid_tokensid.txt

purchasedkeywordid_tokensid.txt

titleid_tokensid.txt

descriptionid_tokensid.txt

userid_profile.txt

Each line of the first four files maps an id to a list of tokens, corresponding to the query, keyword, ad title, and ad description, respectively. In each line, a TAB character separates the id and the token set. A token can basically be a word in a natural language. For anonymity, each token is represented by its hash value. Tokens are delimited by the character ‘|’.

Each line of ‘userid_profile.txt’ is composed of UserID, Gender, and Age, delimited by the TAB character. Note that not every UserID in the training and the testing set will be present in ‘userid_profile.txt’. Each field is described below:

Gender: '1' for male, '2' for female, and '0' for unknown.

Age: '1' for (0, 12], '2' for (12, 18], '3' for (18, 24], '4' for (24, 30], '5' for (30, 40], and '6' for greater than 40.

2.1.2 Example Data point
training.txt

Click Impression    AdURL        AdId      AdvId  Depth Pos  QId       KeyId    TitleId  DescId  UId
0    1   4298118681424644510    7686695 385     3     3  1601       5521     7709     576    490234
0    1   4860571499428580850    21560664    37484     2   2  2255103    317      48989    44771  490234
0    1   9704320783495875564    21748480    36759     3   3  4532751    60721    685038   29681  490234
queryid_tokensid.txt

QId Query
0   12731
1   1545|75|31
2   383
3   518|1996
4   4189|75|31
purchasedkeywordid_tokensid.txt

titleid_tokensid.txt

TitleId Title
0   615|1545|75|31|1|138|1270|615|131
1   466|582|685|1|42|45|477|314
2   12731|190|513|12731|677|183
3   2371|3970|1|2805|4340|3|2914|10640|3688|11|834|3
4   165|134|460|2887|50|2|17527|1|1540|592|2181|3|...
descriptionid_tokensid.txt

DescId  Description
0   1545|31|40|615|1|272|18889|1|220|511|20|5270|1...
1   172|46|467|170|5634|5112|40|155|1965|834|21|41...
2   2672|6|1159|109662|123|49933|160|848|248|207|1...
3   13280|35|1299|26|282|477|606|1|4016|1671|771|1...
4   13327|99|128|494|2928|21|26500|10|11733|10|318
userid_profile.txt

UId Gender  Age
1   1   5
2   2   3
3   1   5
4   1   3
5   2   1
2.2 Mapping the Real-world to a Machine Learning problem
2.2.1 Type of Machine Learning Problem
It is a regression problem as we predicting CTR = #clicks/#impressions

2.2.2 Performance metric
Souce : https://www.kaggle.com/c/kddcup2012-track2#Evaluation
ROC: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/receiver-operating-characteristic-curve-roc-curve-and-auc-1/
