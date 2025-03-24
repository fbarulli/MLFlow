# MLFlow
dvc remote add -d myremote https://dagshub.com/fbarulli/MLFlow.dvc
dvc remote modify myremote --local auth basic
dvc remote modify myremote --local user fbarulli
dvc remote modify myremote --local password dhp_yourDagsHubTokenHere