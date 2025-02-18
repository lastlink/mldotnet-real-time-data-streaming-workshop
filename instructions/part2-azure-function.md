## Publish Azure Function to Predict Fraudulent Transactions
All incoming transactions get evaluated to determine if they are fraudulent or not based on our traine ML.NET Machine Learning Model.
This prediction occurs in an Azure Function that we will go ahead and deploy.

### Prerequisites
- Visual Studio or VS Code

<br/>

<details>
<summary>Deploy from Visual Studio</summary>
  <p>
    
To deploy the Azure Function, please follow the steps listed below:

#### 1. Clone source code
Please clone this repository locally using for example a Git command prompt or Github Desktop.
Open the FraudPredictionFunction solution [here](https://github.com/aslotte/mldotnet-real-time-data-streaming-workshop/tree/master/src/real-time-data-streaming/fraud-prediction-function)

#### 2. Build solution and Publish to Azure
Build the solution and publish the function to your new Function App.

To publish the function:

1. Right click on the solution and select "Publish"
![Publish](https://github.com/aslotte/mldotnet-real-time-data-streaming-workshop/blob/master/instructions/images/publish-function-1.png)

2. Check the radio button "Select Existing" and check "Run from Package File". Click Next.
![Selections](https://github.com/aslotte/mldotnet-real-time-data-streaming-workshop/blob/master/instructions/images/publish-function-2.png)

3. Select your Azure Subscription and navigate to your Function app. Select and click ok
![Subscription](https://github.com/aslotte/mldotnet-real-time-data-streaming-workshop/blob/master/instructions/images/publish-function-3.png)

4. Click **Publish**

5. Click **Yes** if asked to update the functions runtime version.
![upgrade](https://github.com/aslotte/mldotnet-real-time-data-streaming-workshop/blob/master/instructions/images/function-upgrade-runtime.png)

</p>
</details>

<details>
<summary>Deploy from VS Code</summary>
  <p>

To deploy the Azure Function from VS Code, please follow the steps listed below:

#### 1. Install the Azure Functions Extension
In VS Code:
- Select View -> Extensions
- Search for **Azure Functions**
- Install the Azure Functions extension

#### 2. Clone the source repository 
Please clone this repository locally using for example a Git command prompt or Github Desktop.
Open the FraudPredictionFunction solution [here](https://github.com/aslotte/mldotnet-real-time-data-streaming-workshop/tree/master/src/real-time-data-streaming/fraud-prediction-function)

#### 3. Open the solution in VS Code
- Select File -> Open Folder... 
- Navigate to the location of the FraudPredictionFunction

#### 4. Sign-in to Azure
- In the menu to the left, select the Azure symbol (at the bottom of the menu)
- Click "Sign-in to Azure" -> Sign-in to your Azure account

#### 5. Publish to Azure
- In the top left, click on the up-arrow to "Deploy to Function App"
![deployToAzure](https://github.com/aslotte/mldotnet-real-time-data-streaming-workshop/blob/master/instructions/images/publish-function-vs-code-publish.png)
- In the top-middle, select the folder you want to deploy (where the function exists)
- Next, select your Azure Subscription
- Next, select the existing, previously created Function app 
- If prompted to update runtime, select yes
- If prompted to optimize for VS Code, select yes 

</p>
</details>
