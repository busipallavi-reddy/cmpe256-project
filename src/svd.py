import numpy as np
import pandas as pd
import json
from flask import Flask
app = Flask(__name__)

class Recommend():
    def __init__(self,similarity,user,N):
        self.similiarity = similarity
        self.user = user
        self.N = N
        self.threshold = 0.9
        
    def loadData(self,path):
        self.ratings_matrix = pd.read_csv(path, index_col=0)
        self.data = np.mat(self.ratings_matrix)      
        
    def processOriginalData(self,path):
        pageId = []
        description = []
        url = []
        with open(path, 'r') as file:
            lines = file.read().splitlines()
            for line in lines:
                items = line.split(',')
                if items[0] == 'A':
                    pageId.append('X' + str(items[1]))
                    description.append(items[3])
                    url.append(items[4])
        data_table = pd.DataFrame({'Description': description, 'url': url}, index=pageId)
        return data_table
    
    def processSVD(self):
        U, Sigma, VT = np.linalg.svd(self.data, full_matrices=False)
        return U,Sigma,VT
    
    def processSingularValue(self,Sigma):
        square  = 0
        energy = 0
        for si in Sigma:
            square = si*si
            energy = energy+square 
        threshold =energy*self.threshold
        energy = 0
        for i, si in enumerate(Sigma):
            sqaure = si * si
            energy += sqaure
            if energy > threshold:
                break
        return i
    
    def processNewSVD(self,U,Sigma,VT,index):
        return U[:, :index], Sigma[:index], VT[:index, :]
    
    def getUnRatedItems(self):
        return np.nonzero(self.data[self.user, :].A == 0)[1]
    
    def getDiagonalMatrix(self,Sigma_new):
        return np.mat(np.eye(len(Sigma_new)) * Sigma_new)
    
    def getTransformedItems(self,U,diag):
        return self.data.T * U * diag.I
    
    def cosineSimilarity(self,A, B):
        num = float(A.T*B)
        denom = np.linalg.norm(A) * np.linalg.norm(B)
        return 0.5 + 0.5 * (num/denom)
    
    def pearsonCorrelationSimilarity(self,A, B):
        if len(A) < 3: return 1
        else: return 0.5 + 0.5 * np.corrcoef(A, B, rowvar=0)[0][1]
        
    def estimateSVD(self,item,transformItem):
        numberOfItems = np.shape(self.data)[1]
        similarityTotal = 0; ratingSimilarityTotal = 0
        if self.similiarity == "cosine": 
            for j in range(numberOfItems):
                userRating = self.data[self.user, j]
                if not (userRating == 0 or j == item):
                    similarity = self.cosineSimilarity(transformItem[item, :].T, transformItem[j, :].T)
                    similarityTotal += similarity
                    ratingSimilarityTotal += similarity * userRating
            if similarityTotal == 0: 
                return 0
            else: 
                return ratingSimilarityTotal / similarityTotal
        else:
            for j in range(numberOfItems):
                userRating = self.data[self.user, j]
                if not (userRating == 0 or j == item):
                    similarity = self.pearsonCorrelationSimilarity(transformItem[item, :].T, transformItem[j, :].T)
                    similarityTotal += similarity
                    ratingSimilarityTotal += similarity * userRating
            if similarityTotal == 0: 
                return 0
            else: 
                return ratingSimilarityTotal / similarityTotal
    
    
    def predictRecommendation(self,itemToRate,transformItem):
        pageScores = []
        for item in itemToRate:
            predictedScore = self.estimateSVD(item,transformItem)
            pageScores.append((item, predictedScore))
        recommendations = sorted(pageScores, key=lambda jj: jj[1], reverse=True)
        if not self.N:
            return recommendations
        else:
            return recommendations[:self.N]


@app.route('/cosine')
def getResponseForCosine():
	predictedURLs = []
	userID = 200
	predictionNumber = 5
	c_recommend = Recommend("cosine",userID,predictionNumber)
	c_recommend.loadData("./datasets/MS_ratings_matrix.csv")
	U,sigma,VT = c_recommend.processSVD()
	index = c_recommend.processSingularValue(sigma)
	U,sigma,VT = c_recommend.processNewSVD(U,sigma,VT,index)
	diag = c_recommend.getDiagonalMatrix(sigma)
	itemsToRate = c_recommend.getUnRatedItems()
	transforMatrix = c_recommend.getTransformedItems(U,diag)
	recommendations = c_recommend.predictRecommendation(itemsToRate,transforMatrix)
	ids = []
	table = c_recommend.processOriginalData("./datasets/anonymous-msweb.csv")
	print("Top 5 Recommended URLs for userID : ", userID)
	for item, score in recommendations:
	    page_id = c_recommend.ratings_matrix.columns[item]
	    ids.append(page_id)
	    predictedURLs.append(table.loc[page_id][1])
	print(predictedURLs)
	response = app.response_class(response=json.dumps(predictedURLs),status=200,mimetype='application/json')
	return response


@app.route('/pearson')
def getResponseForPearson():
	predictedURLs = []
	userID = 200
	predictionNumber = 5
	p_recommend = Recommend("pearson",userID,predictionNumber)
	p_recommend.loadData("./datasets/MS_ratings_matrix.csv")
	U,sigma,VT = p_recommend.processSVD()
	index = p_recommend.processSingularValue(sigma)
	U,sigma,VT = p_recommend.processNewSVD(U,sigma,VT,index)
	diag = p_recommend.getDiagonalMatrix(sigma)
	itemsToRate = p_recommend.getUnRatedItems()
	transforMatrix = p_recommend.getTransformedItems(U,diag)
	recommendations = p_recommend.predictRecommendation(itemsToRate,transforMatrix)
	ids = []
	table = p_recommend.processOriginalData("./datasets/anonymous-msweb.csv")
	print("Top 5 Recommended URLs for userID : ", userID)
	for item, score in recommendations:
	    page_id = p_recommend.ratings_matrix.columns[item]
	    ids.append(page_id)
	    predictedURLs.append(table.loc[page_id][1])
	print(predictedURLs)
	response = app.response_class(response=json.dumps(predictedURLs),status=200,mimetype='application/json')
	return response


if __name__ == '__main__':
    app.run()