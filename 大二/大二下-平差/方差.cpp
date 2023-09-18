#include<bits/stdc++.h>

using namespace std;

int main(){
	
	int ob[1010];
	int n;
	double sum = 0;
	double average =0;
	double powSum = 0;
	cin>>n;
	for(int i=0; i<n; i++){
		cin >>ob[i];
		sum+=ob[i];
	}
	average = sum/n;
	for(int i=0; i<n; i++){
		powSum += (ob[i]-average)*(ob[i]-average);
	}
	printf("%.6lf", powSum/n);

}
