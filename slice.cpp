#include<cstdio>
#include<algorithm>
#include<string>
#include<cstring>
#include<iostream>
#include<fstream>
using namespace std;
int main(){
	freopen("source.txt","r",stdin);
	//freopen("out.txt","w",stdout);
	ofstream of1,of2;
	of1.open("input1.txt");
	of2.open("input2.txt");
	string str,str2[200];
	int h=0;
	double temp;
	while(cin>>str){
        if(str[0]=='E'){
			for(int i=0;i<6;i++){
				cin>>temp;
				if(i!=5)of1<<temp<<" ";
				else of1<<temp<<endl;
			}
		}
		if(str=="Direct"){
			getline(cin,str);
			for(int i=0;i<3;i++){
				getline(cin,str2[h++]);
			}
		}
	}
	for(int i=0;i<h;i++){
		of2<<str2[i]<<endl;
	}
	return 0;
}
