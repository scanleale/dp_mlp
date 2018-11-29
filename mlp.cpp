/************************************************************************
 *  mlp.cpp - Implements a multi-layer back-propagation neural network
 *  CSCI964/CSCI464 2-Layer MLP
 *  Ver1: Koren Ward - 15 March 2003
 *  Ver2: Koren Ward - 21 July  2003 - Dynamic memory added
 *  Ver3: Koren Ward - 20 March 2005 - Net paramaters in datafile added
 *  Ver4: Your Name -  ?? April 2005 - 3, 4 & 5 layer mlp & test fn added
 *  Copyright - University of Wollongong - 2005
 *************************************************************************/
#include<iostream>
#include<iomanip>
#include<fstream>
#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<ctime>

#include <iterator>        // std::ostream_iterator
#include <vector>          // std::vector
#include <algorithm>       // std::copy

using namespace std;

const int MAXN = 50;       // Max neurons in any layer
const int MAXPATS = 50000;  // Max training patterns

// mlp paramaters
long NumIts;    // Max training iterations
int NumHN;      // Number of hidden layers
int NumHN1;     // Number of neurons in hidden layer 1
int NumHN2;     // Number of neurons in hidden layer 2
int NumHN3;     // Number of neurons in hidden layer 3
int NumHN4;     // Number of neurons in hidden layer 4
float LrnRate;  // Learning rate
float Mtm1;     // Momentum(t-1)
float Mtm2;     // Momentum(t-2)
float ObjErr;   // Objective error
int Ordering;   // Type of patterns order

// standard deviation normalization
int Norm;       // Normalization sign
vector<float> mean;
vector<float> deviation;

std::vector<int> RdmLists; // index array;

// mlp weights
float **w1, **w11, **w111;    // input and 1st or output layer wts
float **w2, **w22, **w222;    // 1st and 2nd or output layer wts, one hidden
float **w3, **w33, **w333;    // 2nd and 3rd or output layer wts, two hidden
float **w4, **w44, **w444;    // 3rd and 4th or output layer wts, three hidden

// reorder training pattern randomly
void RandomPattern(vector<int> &x);      
// swap two training patterns randomly                                  
void RandomSwap(vector<int> &x);          
// standard deviation normalization                                 
void GetMD(float **x, vector<float> &m, vector<float> &d, int NumPats);    
// normalize data
void Normalize(float **x, vector<float> &m, vector<float> &d, int NumPats);
// prepare data according to para (ordering)
void PrepareTraPats(int ItCnt);                                            

// 3 layers (1 hidden layers)
void TrainNet3(float **x, float **d, int NumIPs, int NumOPs, int NumPats); 
// 4 layers (2 hidden layers)
void TrainNet4(float **x, float **d, int NumIPs, int NumOPs, int NumPats); 
// 5 layers (3 hidden layers)
void TrainNet5(float **x, float **d, int NumIPs, int NumOPs, int NumPats); 

// test the training results
void TestNet(float **x, float **d, int NumIPs, int NumOPs, int NumPats);   

float **Aloc2DAry(int m, int n);
void Free2DAry(float **Ary2D, int n);

int main() {
  ifstream fin;
  int i, j, NumIPs, NumOPs, NumTrnPats, NumTstPats;
  char Line[500], Tmp[20], FName[20];
  cout << "Enter data filename: ";
  cin >> FName;
  cin.ignore();
  fin.open(FName);
  if (!fin.good()) {
    cout << "File not found!\n";
    exit(1);
  }
  //read data specs...
  do {
    fin.getline(Line, 500);
  } while (Line[0] == ';'); //eat comments
  sscanf(Line, "%s%d", Tmp, &NumIPs);
  fin >> Tmp >> NumOPs;
  fin >> Tmp >> NumTrnPats;
  fin >> Tmp >> NumTstPats;
  fin >> Tmp >> NumIts;
  fin >> Tmp >> NumHN;
  i = NumHN;
  if (i-- > 0)
    fin >> Tmp >> NumHN1;
  if (i-- > 0)
    fin >> Tmp >> NumHN2;
  if (i-- > 0)
    fin >> Tmp >> NumHN3;
  if (i-- > 0)
    fin >> Tmp >> NumHN4;
  fin >> Tmp >> LrnRate;
  fin >> Tmp >> Mtm1;
  fin >> Tmp >> Mtm2;
  fin >> Tmp >> ObjErr;
  fin >> Tmp >> Ordering;
  //fin >> Tmp >> Norm;
  
  if (NumIPs < 1 || NumIPs > MAXN || NumOPs < 1 || NumOPs > MAXN || NumTrnPats < 1 
      || NumTrnPats > MAXPATS|| NumTrnPats < 1 || NumTrnPats > MAXPATS || NumIts < 1 
	  || NumIts > 20e6 || NumHN1 < 0 || NumHN1 > 50 || LrnRate < 0 || LrnRate > 1 
	  || Mtm1 < 0 || Mtm1 > 10 || Mtm2 < 0 || Mtm2 > 10 || ObjErr < 0 || ObjErr > 10
      || NumHN2 < 0 || NumHN2 > 50 || NumHN3 < 0 || NumHN3 > 50 || NumHN4 < 0 
	  || NumHN4 > 50 || NumHN > 4 || NumHN < 1 || (Norm!=0 && Norm!=1)) {
    cout << "Invalid specs in data file!\n"; exit(1);
  }
  float **IPTrnData = Aloc2DAry(NumTrnPats, NumIPs);
  float **OPTrnData = Aloc2DAry(NumTrnPats, NumOPs);
  float **IPTstData = Aloc2DAry(NumTstPats, NumIPs);
  float **OPTstData = Aloc2DAry(NumTstPats, NumOPs);
  for (i = 0; i < NumTrnPats; i++) {
    for (j = 0; j < NumIPs; j++)
      fin >> IPTrnData[i][j];  // read trainning data
    for (j = 0; j < NumOPs; j++)
      fin >> OPTrnData[i][j];  // read lable of trainning data
  }
  for (i = 0; i < NumTstPats; i++) {
    for (j = 0; j < NumIPs; j++)
      fin >> IPTstData[i][j];  // read testing data
    for (j = 0; j < NumOPs; j++)
      fin >> OPTstData[i][j];  // read lable of testing data
  }
  fin.close();

  if(Norm==1){                 // normalize data (training & testing patterns)
	mean.resize(NumIPs, 0.0);
	deviation.resize(NumIPs, 0.0);
    GetMD(IPTrnData, mean, deviation, NumTrnPats);     // get mean & deviation
    Normalize(IPTrnData, mean, deviation, NumTrnPats); // use the vale to normalize
    Normalize(IPTstData, mean, deviation, NumTstPats); // use the same vale to normalize
  }
  
  // initial index array
  RdmLists.resize(NumTrnPats);
  for (int j = 0; j < NumTrnPats; ++j)
    RdmLists[j] = j;
  switch (NumHN){
    case 1:
	  TrainNet3(IPTrnData, OPTrnData, NumIPs, NumOPs, NumTrnPats); break;
    case 2:
	  TrainNet4(IPTrnData, OPTrnData, NumIPs, NumOPs, NumTrnPats); break;
	case 3:
	  TrainNet5(IPTrnData, OPTrnData, NumIPs, NumOPs, NumTrnPats); break;
    default: break;	  
  }
  
  TestNet(IPTstData, OPTstData, NumIPs, NumOPs, NumTstPats);

  Free2DAry(IPTrnData, NumTrnPats);
  Free2DAry(OPTrnData, NumTrnPats);
  Free2DAry(IPTstData, NumTstPats);
  Free2DAry(OPTstData, NumTstPats);
  cout << "End of program.\n";
  //system("PAUSE");
  return 0;
}

/*
 * data preparation
 */
void PrepareTraPats(int ItCnt){
  switch (Ordering){
	case 0: break;
	case 1: 
	  RandomPattern(RdmLists);
	  break;
	case 2: 
	  if(ItCnt==0) RandomPattern(RdmLists); // the first round
      else RandomSwap(RdmLists);            // from the second round
	  break;
	default:
	  // Ordering > 2
	  RandomPattern(RdmLists); // reorder all and pick the Nth from the beginning
  }
}

/*
 * implementation for 3 layers (1 hidden layer)
 */
void TrainNet3(float **x, float **d, int NumIPs, int NumOPs, int NumPats) {
  float *h1 = new float[NumHN1];         // O/Ps of hidden layer
  float *y = new float[NumOPs];          // O/P of Net
  float *ad1 = new float[NumHN1];        // HN1 back prop errors
  float *ad2 = new float[NumOPs];        // O/P back prop errors
  float PatErr, MinErr, AveErr, MaxErr;  // Pattern errors
  int p, i, j;     // for loops indexes
  long ItCnt = 0;  // Iteration counter
  long NumErr = 0; // Error counter (added for spiral problem)

  cout << "NetArch: IP:" << NumIPs << " H1:" << NumHN1 << " OP:" << NumOPs << endl;
  cout << "Params: LrnRate: " << LrnRate << " Mtm1: " << Mtm1 << " Mtm2: " << Mtm2 
    << " Ordering: " << Ordering << " Norm: " << Norm << endl;
  cout << "Training mlp for " << NumIts << " iterations:" << endl;
  cout << setprecision(6) << setw(7) << "#" << setw(12) << "MinErr" << setw(12) 
    << "AveErr" << setw(12) << "MaxErr" << setw(12) << "\%PcntErr" << endl;
  // Allocate memory for weights
  w1 = Aloc2DAry(NumIPs, NumHN1); // 1st layer wts
  w11 = Aloc2DAry(NumIPs, NumHN1);
  w111 = Aloc2DAry(NumIPs, NumHN1);

  w2 = Aloc2DAry(NumHN1, NumOPs); // 2nd layer wts
  w22 = Aloc2DAry(NumHN1, NumOPs);
  w222 = Aloc2DAry(NumHN1, NumOPs);

  // Init wts between -0.5 and +0.5
  srand(time(0));
  for (i = 0; i < NumIPs; i++)
    for (j = 0; j < NumHN1; j++)
      w1[i][j] = w11[i][j] = w111[i][j] = float(rand()) / RAND_MAX - 0.5;
  for (i = 0; i < NumHN1; i++)
    for (j = 0; j < NumOPs; j++)
      w2[i][j] = w22[i][j] = w222[i][j] = float(rand()) / RAND_MAX - 0.5;

  for (;;) {                                               // Main learning loop
    int thePattern = 0;                                    // training from the pattern
	PrepareTraPats(ItCnt);
    if (Ordering > 2 && ItCnt > 0) {
      for (int n = 0; n < (Ordering - 1); ++n) {  
        for (i = 0; i < NumHN1; i++) {                     // Cal O/P of hidden layer 1
          float in = 0;
          for (j = 0; j < NumIPs; j++)
            in += w1[j][i] * x[RdmLists[n]][j];
          h1[i] = (float) (1.0 / (1.0 + exp(double(-in))));// Sigmoid
        }
        for (i = 0; i < NumOPs; i++) {                     // Cal O/P of output layer
          float in = 0;
          for (j = 0; j < NumHN1; j++) {
            in += w2[j][i] * h1[j];
          }
          y[i] = (float) (1.0 / (1.0 + exp(double(-in)))); // Sigmoid
        }
        int isErr = 0;                                     // Cal error for this pattern
        for (i = 0; i < NumOPs; i++) {
		  isErr += ((y[i] < 0.5 && d[RdmLists[n]][i] >= 0.5) 
		           || (y[i] >= 0.5 && d[RdmLists[n]][i] < 0.5));
        }
        if (isErr > 0) {
          thePattern = n;
          break;
        }
      }
    }
	
    MinErr = 3.4e38; AveErr = 0;  MaxErr = -3.4e38;  NumErr = 0;
    int rIdx; // the actual index associated with a random index
    for (p = thePattern; p < NumPats; p++) {
      rIdx = RdmLists[p];
      // Cal neural network output
      for (i = 0; i < NumHN1; i++) {                       // Cal O/P of hidden layer 1
        float in = 0;
        for (j = 0; j < NumIPs; j++)
          in += w1[j][i] * x[rIdx][j];  
        h1[i] = (float) (1.0 / (1.0 + exp(double(-in))));  // Sigmoid
      }
      for (i = 0; i < NumOPs; i++) {                       // Cal O/P of output layer
        float in = 0;
        for (j = 0; j < NumHN1; j++) {
          in += w2[j][i] * h1[j];
        }
        y[i] = (float) (1.0 / (1.0 + exp(double(-in))));   // Sigmoid      
	  }
      // Cal error for this pattern
      PatErr = 0.0;
      for (i = 0; i < NumOPs; i++) {
        float err = y[i] - d[rIdx][i];                     // actual-desired O/P
        if (err > 0) PatErr += err; else PatErr -= err;    // abs
        
//        float err=0.5*(y[i] - d[rIdx][i])*(y[i] - d[rIdx][i]); // actual-desired O/P
//
//        PatErr+=err;
        
		//added for binary classification problem
        NumErr += ((y[i] < 0.5 && d[rIdx][i] >= 0.5) || (y[i] >= 0.5 && d[rIdx][i] < 0.5)); 
      }
      if (PatErr < MinErr) MinErr = PatErr;
      if (PatErr > MaxErr) MaxErr = PatErr;
      AveErr += PatErr;

      // Learn pattern with back propagation
	  float tmp;
      for (i = 0; i < NumOPs; i++) {                       // Modify layer 2 wts
        ad2[i] = (d[rIdx][i] - y[i]) * y[i] * (1.0 - y[i]);
        for (j = 0; j < NumHN1; j++) {
		  tmp= w2[j][i];
          w2[j][i] += LrnRate * h1[j] * ad2[i] 
		      + Mtm1 * (w2[j][i] - w22[j][i])
              + Mtm2 * (w22[j][i] - w222[j][i]);
          w222[j][i] = w22[j][i];
          w22[j][i] = tmp;
        }
      }
	  
      for (i = 0; i < NumHN1; i++) {                       // Modify layer 1 wts
        float err = 0.0;
        for (j = 0; j < NumOPs; j++)
          err += ad2[j] * w22[i][j];
        ad1[i] = err * h1[i] * (1.0 - h1[i]);
        for (j = 0; j < NumIPs; j++) {
		  tmp= w1[j][i];
          w1[j][i] += LrnRate * x[rIdx][j] * ad1[i] 
		      + Mtm1 * (w1[j][i] - w11[j][i])
              + Mtm2 * (w11[j][i] - w111[j][i]);
          w111[j][i] = w11[j][i];
          w11[j][i] = tmp;
        }
      }
      // when Ordering>2 and not the first epoch then exit after training one pattern???
      if (Ordering > 2 && ItCnt > 0) break;
    } // end for each pattern
    ItCnt++;
    float PcntErr = 0.0;
    if (Ordering <= 2 || ItCnt == 1) {
      AveErr /= NumPats;
      PcntErr = NumErr / float(NumPats) * 100.0;
    } else { 
      AveErr /= 1; //NumPats is 1, if ordering>2 && not the first round;
      PcntErr = NumErr / float(1) * 100.0;
    }
    cout.setf(ios::fixed | ios::showpoint);
    cout << setprecision(6) << setw(6) << ItCnt << ": " << setw(12) << MinErr << setw(12) 
	    << AveErr << setw(12) << MaxErr << setw(12) << PcntErr << endl;
    if ((AveErr <= ObjErr) || (ItCnt == NumIts))
      break;
  } // end main learning loop
    // Free memory
  delete h1;
  delete y;
  delete ad1;
  delete ad2;
}

/*
 * implementation for 4 layers (2 hidden layers)
 */
void TrainNet4(float **x, float **d, int NumIPs, int NumOPs, int NumPats) {
  float *h1 = new float[NumHN1];         // O/Ps of hidden layer 1
  float *h2 = new float[NumHN2];         // O/Ps of hidden layer 2
  float *y = new float[NumOPs];          // O/P of Net
  float *ad1 = new float[NumHN1];        // HN1 back prop errors
  float *ad2 = new float[NumHN2];        // HN2 back prop errors
  float *ad3 = new float[NumOPs];        // O/P back prop errors
  float PatErr, MinErr, AveErr, MaxErr;  // Pattern errors
  int p, i, j;     // for loops indexes
  long ItCnt = 0;  // Iteration counter
  long NumErr = 0; // Error counter (added for spiral problem)

  cout << "NetArch: IP:" << NumIPs << " H1:" << NumHN1 << " H2:" << NumHN2 << " OP:" 
    << NumOPs << endl;
  cout << "Params: LrnRate: " << LrnRate << " Mtm1: " << Mtm1 << " Mtm2: " << Mtm2 
    << " Ordering: " << Ordering << " Norm: " << Norm << endl;
  cout << "Training mlp for " << NumIts << " iterations:" << endl;
  cout << setprecision(6) << setw(7) << "#" << setw(12) << "MinErr" << setw(12) 
    << "AveErr" << setw(12) << "MaxErr" << setw(12) << "\%PcntErr" << endl;
  // Allocate memory for weights
  w1 = Aloc2DAry(NumIPs, NumHN1);        // input to 1st layer wts
  w11 = Aloc2DAry(NumIPs, NumHN1);
  w111 = Aloc2DAry(NumIPs, NumHN1);

  w2 = Aloc2DAry(NumHN1, NumHN2);        // 1st to 2nd layer wts
  w22 = Aloc2DAry(NumHN1, NumHN2);
  w222 = Aloc2DAry(NumHN1, NumHN2);

  w3 = Aloc2DAry(NumHN2, NumOPs);        // 2nd to output layer wts
  w33 = Aloc2DAry(NumHN2, NumOPs);
  w333 = Aloc2DAry(NumHN2, NumOPs);

  // Init wts between -0.5 and +0.5
  srand(time(0));
  for (i = 0; i < NumIPs; i++)
    for (j = 0; j < NumHN1; j++)
      w1[i][j] = w11[i][j] = w111[i][j] = float(rand()) / RAND_MAX - 0.5;
  for (i = 0; i < NumHN1; i++)
    for (j = 0; j < NumHN2; j++)
      w2[i][j] = w22[i][j] = w222[i][j] = float(rand()) / RAND_MAX - 0.5;
  for (i = 0; i < NumHN2; i++)
    for (j = 0; j < NumOPs; j++)
      w3[i][j] = w33[i][j] = w333[i][j] = float(rand()) / RAND_MAX - 0.5;

  for (;;) {                                               // Main learning loop
    int thePattern = 0;                                    // training from the pattern
	PrepareTraPats(ItCnt);
    if (Ordering > 2 && ItCnt > 0) {
      for (int n = 0; n < (Ordering - 1); ++n) {
        for (i = 0; i < NumHN1; i++) {                     // Cal O/P of hidden layer 1
          float in = 0;
          for (j = 0; j < NumIPs; j++)
            in += w1[j][i] * x[RdmLists[n]][j];
          h1[i] = (float) (1.0 / (1.0 + exp(double(-in))));// Sigmoid
        }
        for (i = 0; i < NumHN2; i++) {                     // Cal O/P of hidden layer 2
          float in = 0;
          for (j = 0; j < NumHN1; j++)
            in += w2[j][i] * h1[j];
          h2[i] = (float) (1.0 / (1.0 + exp(double(-in))));// Sigmoid
        }
        for (i = 0; i < NumOPs; i++) {                     // Cal O/P of output layer
          float in = 0;
          for (j = 0; j < NumHN2; j++) {
            in += w3[j][i] * h2[j];
          }
          y[i] = (float) (1.0 / (1.0 + exp(double(-in)))); // Sigmoid
        }
        // Cal error for this pattern
        int isErr = 0;
        for (i = 0; i < NumOPs; i++) {
		  isErr += ((y[i] < 0.5 && d[RdmLists[n]][i] >= 0.5) 
		          || (y[i] >= 0.5 && d[RdmLists[n]][i] < 0.5));
        }
        if (isErr > 0) {
          thePattern = n;
          break;
        }
      }
    }  

    MinErr = 3.4e38; AveErr = 0; MaxErr = -3.4e38; NumErr = 0;
    int rIdx; // the actual index associated with a random index
    for (p = thePattern; p < NumPats; p++) { 
      rIdx = RdmLists[p]; 
      // Cal neural network output
      for (i = 0; i < NumHN1; i++) {                       // Cal O/P of hidden layer 1
        float in = 0;
        for (j = 0; j < NumIPs; j++)
          in += w1[j][i] * x[rIdx][j];
        h1[i] = (float) (1.0 / (1.0 + exp(double(-in))));  // Sigmoid
      }
      for (i = 0; i < NumHN2; i++) {                       // Cal O/P of hidden layer 2
        float in = 0;
        for (j = 0; j < NumHN1; j++)
          in += w2[j][i] * h1[j];
        h2[i] = (float) (1.0 / (1.0 + exp(double(-in))));  // Sigmoid
      }
      for (i = 0; i < NumOPs; i++) {                       // Cal O/P of output layer
        float in = 0;
        for (j = 0; j < NumHN2; j++) {
          in += w3[j][i] * h2[j];
        }
        y[i] = (float) (1.0 / (1.0 + exp(double(-in)))); // Sigmoid
      }
      // Cal error for this pattern
      PatErr = 0.0;
      for (i = 0; i < NumOPs; i++) {
        float err = y[i] - d[rIdx][i]; // actual-desired O/P
        if (err > 0) PatErr += err; else PatErr -= err;
		//added for binary classification problem
        NumErr += ((y[i] < 0.5 && d[rIdx][i] >= 0.5) || (y[i] >= 0.5 && d[rIdx][i] < 0.5)); 
      }
      if (PatErr < MinErr) MinErr = PatErr;
      if (PatErr > MaxErr) MaxErr = PatErr;
      AveErr += PatErr;

      // Learn pattern with back propagation
	  float tmp;
      for (i = 0; i < NumOPs; i++) {                       // Modify output layer 2 wts
        ad3[i] = (d[rIdx][i] - y[i]) * y[i] * (1.0 - y[i]);// delta[i]=(t[i]-y[i])*y[i]*(1-y[i])
        for (j = 0; j < NumHN2; j++) {
		  tmp= w3[j][i];
          w3[j][i] += LrnRate * h2[j] * ad3[i]             // w[j][i]=w[j][i]-delta[i]
		    + Mtm1 * (w3[j][i] - w33[j][i]) 
			+ Mtm2 * (w33[j][i] - w333[j][i]);
          w333[j][i] = w33[j][i];
          w33[j][i] = tmp;
        }
      }
      for (i = 0; i < NumHN2; i++) {                       // Modify layer 2-1 wts
        float err = 0.0;
        for (j = 0; j < NumOPs; j++)
          err += ad3[j] * w33[i][j];
        ad2[i] = err * h2[i] * (1.0 - h2[i]);
        for (j = 0; j < NumHN1; j++) {
		  tmp= w2[j][i];
          w2[j][i] += LrnRate * h1[j] * ad2[i] 
		    + Mtm1 * (w2[j][i] - w22[j][i])
			+ Mtm2 * (w22[j][i] - w222[j][i]);
          w222[j][i] = w22[j][i];
          w22[j][i] = tmp;
        }
      }
      for (i = 0; i < NumHN1; i++) {                       // Modify layer 1-input wts
        float err = 0.0;
        for (j = 0; j < NumHN2; j++)
          err += ad2[j] * w22[i][j];
        ad1[i] = err * h1[i] * (1.0 - h1[i]);
        for (j = 0; j < NumIPs; j++) {
		  tmp= w1[j][i];
          w1[j][i] += LrnRate * x[rIdx][j] * ad1[i] 
		    + Mtm1 * (w1[j][i] - w11[j][i]) 
			+ Mtm2 * (w11[j][i] - w111[j][i]);
          w111[j][i] = w11[j][i];
          w11[j][i] = tmp;
        }
      }
      // when Ordering>2 and not the first epoch then exit after training one pattern???
      if (Ordering > 2 && ItCnt > 0)
        break;
    } // end for each pattern
	//cout << endl;
    ItCnt++;
    float PcntErr = 0.0;
    if (Ordering <= 2 || ItCnt == 1) {
      AveErr /= NumPats;
      PcntErr = NumErr / float(NumPats) * 100.0;
    } else {
      AveErr /= 1; //NumPats is 1, if ordering>2 && not the first round;
      PcntErr = NumErr / float(1) * 100.0;
    }
    cout.setf(ios::fixed | ios::showpoint);
    cout << setprecision(6) << setw(6) << ItCnt << ": " << setw(12) << MinErr << setw(12) 
	    << AveErr << setw(12) << MaxErr << setw(12) << PcntErr << endl;
	
    if ((AveErr <= ObjErr) || (ItCnt == NumIts))
      break;
  } // end main learning loop
    // Free memory
  delete h1;
  delete h2;
  delete y;
  delete ad1;
  delete ad2;
  delete ad3;
}

/*
 * implementation for 5 layers (3 hidden layers)
 */
void TrainNet5(float **x, float **d, int NumIPs, int NumOPs, int NumPats) {
  float *h1 = new float[NumHN1];         // O/Ps of hidden layer 1
  float *h2 = new float[NumHN2];         // O/Ps of hidden layer 2
  float *h3 = new float[NumHN3];         // O/Ps of hidden layer 3
  float *y = new float[NumOPs];          // O/P of Net
  float *ad1 = new float[NumHN1];        // HN1 back prop errors
  float *ad2 = new float[NumHN2];        // HN2 back prop errors
  float *ad3 = new float[NumHN3];        // HN2 back prop errors
  float *ad4 = new float[NumOPs];        // O/P back prop errors
  float PatErr, MinErr, AveErr, MaxErr;  // Pattern errors
  int p, i, j;     // for loops indexes
  long ItCnt = 0;  // Iteration counter
  long NumErr = 0; // Error counter (added for spiral problem)

  cout << "NetArch: IP:" << NumIPs << " H1:" << NumHN1 << " H2:" << NumHN2 << " H3:" 
    << NumHN3 << " OP:" << NumOPs << endl;
  cout << "Params: LrnRate: " << LrnRate << " Mtm1: " << Mtm1 << " Mtm2: " << Mtm2 
    << " Ordering: " << Ordering << " Norm: " << Norm << endl;
  cout << "Training mlp for " << NumIts << " iterations:" << endl;
  cout << setprecision(6) << setw(7) << "#" << setw(12) << "MinErr" << setw(12) 
    << "AveErr" << setw(12) << "MaxErr" << setw(12) << "\%PcntErr" << endl;
  // Allocate memory for weights
  w1 = Aloc2DAry(NumIPs, NumHN1); // input to 1st layer wts
  w11 = Aloc2DAry(NumIPs, NumHN1);
  w111 = Aloc2DAry(NumIPs, NumHN1);

  w2 = Aloc2DAry(NumHN1, NumHN2); // 1st to 2nd layer wts
  w22 = Aloc2DAry(NumHN1, NumHN2);
  w222 = Aloc2DAry(NumHN1, NumHN2);

  w3 = Aloc2DAry(NumHN2, NumHN3); // 2nd to output layer wts
  w33 = Aloc2DAry(NumHN2, NumHN3);
  w333 = Aloc2DAry(NumHN2, NumHN3);

  w4 = Aloc2DAry(NumHN3, NumOPs); // 3nd to output layer wts
  w44 = Aloc2DAry(NumHN3, NumOPs);
  w444 = Aloc2DAry(NumHN3, NumOPs);
  
  // Init wts between -0.5 and +0.5
  srand(time(0));
  for (i = 0; i < NumIPs; i++)
    for (j = 0; j < NumHN1; j++)
      w1[i][j] = w11[i][j] = w111[i][j] = float(rand()) / RAND_MAX - 0.5;
  for (i = 0; i < NumHN1; i++)
    for (j = 0; j < NumHN2; j++)
      w2[i][j] = w22[i][j] = w222[i][j] = float(rand()) / RAND_MAX - 0.5;
  for (i = 0; i < NumHN2; i++)
    for (j = 0; j < NumHN3; j++)
      w3[i][j] = w33[i][j] = w333[i][j] = float(rand()) / RAND_MAX - 0.5;
  for (i = 0; i < NumHN3; i++)
    for (j = 0; j < NumOPs; j++)
      w4[i][j] = w44[i][j] = w444[i][j] = float(rand()) / RAND_MAX - 0.5;
  
  for (;;) {  // Main learning loop
    int thePattern = 0;                                    // training from the pattern
	PrepareTraPats(ItCnt);
    if (Ordering > 2 && ItCnt > 0) {
      for (int n = 0; n < (Ordering - 1); ++n) {
        for (i = 0; i < NumHN1; i++) {                     // Cal O/P of hidden layer 1
          float in = 0;
          for (j = 0; j < NumIPs; j++)
            in += w1[j][i] * x[RdmLists[n]][j];
          h1[i] = (float) (1.0 / (1.0 + exp(double(-in))));// Sigmoid
        }
        for (i = 0; i < NumHN2; i++) {                     // Cal O/P of hidden layer 2
          float in = 0;
          for (j = 0; j < NumHN1; j++)
            in += w2[j][i] * h1[j];
          h2[i] = (float) (1.0 / (1.0 + exp(double(-in))));// Sigmoid
        }
        for (i = 0; i < NumHN3; i++) {                     // Cal O/P of hidden layer 3
          float in = 0;
          for (j = 0; j < NumHN2; j++)
            in += w3[j][i] * h2[j];
          h3[i] = (float) (1.0 / (1.0 + exp(double(-in))));// Sigmoid
        }
        for (i = 0; i < NumOPs; i++) {                     // Cal O/P of output layer
          float in = 0;
          for (j = 0; j < NumHN3; j++) {
            in += w4[j][i] * h3[j];
          }
          y[i] = (float) (1.0 / (1.0 + exp(double(-in)))); // Sigmoid
        }
        // Cal error for this pattern
        int isErr = 0;
        for (i = 0; i < NumOPs; i++) {
		  isErr += ((y[i] < 0.5 && d[RdmLists[n]][i] >= 0.5) 
		        || (y[i] >= 0.5 && d[RdmLists[n]][i] < 0.5));
        }
        if (isErr > 0) {
          thePattern = n;
          break;
        }
      }
    }

    MinErr = 3.4e38; AveErr = 0; MaxErr = -3.4e38; NumErr = 0;
    int rIdx; // the actual index associated with a random index
    for (p = thePattern; p < NumPats; p++) {
      rIdx = RdmLists[p];            
      // Cal neural network output
      for (i = 0; i < NumHN1; i++) {                       // Cal O/P of hidden layer 1
        float in = 0;
        for (j = 0; j < NumIPs; j++)
          in += w1[j][i] * x[rIdx][j];
        h1[i] = (float) (1.0 / (1.0 + exp(double(-in))));  // Sigmoid
      }
      for (i = 0; i < NumHN2; i++) {                       // Cal O/P of hidden layer 2
        float in = 0;
        for (j = 0; j < NumHN1; j++)
          in += w2[j][i] * h1[j];
        h2[i] = (float) (1.0 / (1.0 + exp(double(-in))));  // Sigmoid
      }
      for (i = 0; i < NumHN3; i++) {                       // Cal O/P of hidden layer 3
        float in = 0;
        for (j = 0; j < NumHN2; j++)
          in += w3[j][i] * h2[j];
        h3[i] = (float) (1.0 / (1.0 + exp(double(-in))));  // Sigmoid
      }
      for (i = 0; i < NumOPs; i++) {                       // Cal O/P of output layer
        float in = 0;
        for (j = 0; j < NumHN3; j++) {
          in += w4[j][i] * h3[j];
        }
        y[i] = (float) (1.0 / (1.0 + exp(double(-in))));   // Sigmoid
      }
      // Cal error for this pattern
      PatErr = 0.0;
      for (i = 0; i < NumOPs; i++) {
        float err = y[i] - d[rIdx][i]; // actual-desired O/P
        if (err > 0) PatErr += err; else PatErr -= err;
		//added for binary classification problem
        NumErr += ((y[i] < 0.5 && d[rIdx][i] >= 0.5) || (y[i] >= 0.5 && d[rIdx][i] < 0.5)); 
      }
      if (PatErr < MinErr) MinErr = PatErr;
      if (PatErr > MaxErr) MaxErr = PatErr;
      AveErr += PatErr;

      // Learn pattern with back propagation
	  float tmp;
      for (i = 0; i < NumOPs; i++) {                       // Modify output-3 wts
        ad4[i] = (d[rIdx][i] - y[i]) * y[i] * (1.0 - y[i]);
        for (j = 0; j < NumHN3; j++) {
		  tmp= w4[j][i];
          w4[j][i] += LrnRate * h3[j] * ad4[i] + Mtm1 * (w4[j][i] - w44[j][i])
              + Mtm2 * (w44[j][i] - w444[j][i]);
          w444[j][i] = w44[j][i];
          w44[j][i] = tmp;
        }
      }
      for (i = 0; i < NumHN3; i++) {                       // Modify layer 3-2 wts
        float err = 0.0;
        for (j = 0; j < NumOPs; j++)
          err += ad4[j] * w44[i][j];
        ad3[i] = err * h3[i] * (1.0 - h3[i]);
        for (j = 0; j < NumHN2; j++) {
		  tmp= w3[j][i];
          w3[j][i] += LrnRate * h2[j] * ad3[i] 
		      + Mtm1 * (w3[j][i] - w33[j][i])
              + Mtm2 * (w33[j][i] - w333[j][i]);
          w333[j][i] = w33[j][i];
          w33[j][i] = tmp;
        }
      }
      for (i = 0; i < NumHN2; i++) {                       // Modify layer 2-1 wts
        float err = 0.0;
        for (j = 0; j < NumHN3; j++)
          err += ad3[j] * w33[i][j];
        ad2[i] = err * h2[i] * (1.0 - h2[i]);
        for (j = 0; j < NumHN1; j++) {
		  tmp= w2[j][i];
          w2[j][i] += LrnRate * h1[j] * ad2[i] 
		      + Mtm1 * (w2[j][i] - w22[j][i])
              + Mtm2 * (w22[j][i] - w222[j][i]);
          w222[j][i] = w22[j][i];
          w22[j][i] = tmp;
        }
      }
      for (i = 0; i < NumHN1; i++) {                       // Modify layer 1-input wts
        float err = 0.0;
        for (j = 0; j < NumHN2; j++)
          err += ad2[j] * w22[i][j];
        ad1[i] = err * h1[i] * (1.0 - h1[i]);
        for (j = 0; j < NumIPs; j++) {
		  tmp= w1[j][i];
          w1[j][i] += LrnRate * x[rIdx][j] * ad1[i] 
		      + Mtm1 * (w1[j][i] - w11[j][i])
              + Mtm2 * (w11[j][i] - w111[j][i]);
          w111[j][i] = w11[j][i];
          w11[j][i] = tmp;
        }
      }
      // when Ordering>2 and not the first epoch then exit after training one pattern???
      if (Ordering > 2 && ItCnt > 0)
        break;
    } // end for each pattern
    ItCnt++;
    float PcntErr = 0.0;
    if (Ordering <= 2 || ItCnt == 1) {
      AveErr /= NumPats;
      PcntErr = NumErr / float(NumPats) * 100.0;
    } else {
      AveErr /= 1; //NumPats is 1, if oredering>2 && not the first round;
      PcntErr = NumErr / float(1) * 100.0; 
    }
    cout.setf(ios::fixed | ios::showpoint);
    cout << setprecision(6) << setw(6) << ItCnt << ": " << setw(12) << MinErr << setw(12) 
	    << AveErr << setw(12) << MaxErr << setw(12) << PcntErr << endl;
    if ((AveErr <= ObjErr) || (ItCnt == NumIts))
      break;
  } // end main learning loop
    // Free memory
  delete h1;
  delete h2;
  delete h3;
  delete y;
  delete ad1;
  delete ad2;
  delete ad3;
  delete ad4;
}

void TestNet(float **x, float **d, int NumIPs, int NumOPs, int NumPats) {
  float PatErr, MinErr, AveErr, MaxErr;  // Pattern errors
  int p, i, j;                           // for loops indexes
  long ItCnt = 0;                        // Iteration counter
  long NumErr = 0;                       // Error counter (added for spiral problem)
  MinErr = 3.4e38;
  AveErr = 0;
  MaxErr = -3.4e38;
  NumErr = 0;
  for (p = 0; p < NumPats; p++) {
    if (NumHN == 1) {
      float *h1 = new float[NumHN1];                       // O/Ps of hidden layer 1
      float *y = new float[NumOPs];                        // O/P of Net
      for (i = 0; i < NumHN1; i++) {                       // Cal O/P of hidden layer 1
        float in = 0;
        for (j = 0; j < NumIPs; j++)
          in += w1[j][i] * x[p][j];
        h1[i] = (float) (1.0 / (1.0 + exp(double(-in))));  // Sigmoid
      }
      for (i = 0; i < NumOPs; i++) {                       // Cal O/P of output layer
        float in = 0;
        for (j = 0; j < NumHN1; j++) {
          in += w2[j][i] * h1[j];
        }
        y[i] = (float) (1.0 / (1.0 + exp(double(-in))));   // Sigmoid
      }
      // Cal error for this pattern
      float PatErr = 0.0;
      for (int i = 0; i < NumOPs; i++) {
        float err = y[i] - d[p][i]; // actual-desired O/P
        if (err > 0) PatErr += err; else PatErr -= err;
		//added for binary classification problem
        NumErr += ((y[i] < 0.5 && d[p][i] >= 0.5) || (y[i] >= 0.5 && d[p][i] < 0.5)); 
      }
      if (PatErr < MinErr) MinErr = PatErr;
      if (PatErr > MaxErr) MaxErr = PatErr;
      AveErr += PatErr;
      delete h1;
      delete y;
    } else if (NumHN == 2) {
      float *h1 = new float[NumHN1];                       // O/Ps of hidden layer 1
      float *h2 = new float[NumHN2];                       // O/Ps of hidden layer 2
      float *y = new float[NumOPs];                        // O/P of Net
      for (i = 0; i < NumHN1; i++) {                       // Cal O/P of hidden layer 1
        float in = 0;
        for (j = 0; j < NumIPs; j++)
          in += w1[j][i] * x[p][j];
        h1[i] = (float) (1.0 / (1.0 + exp(double(-in))));  // Sigmoid
      }
      for (i = 0; i < NumHN2; i++) {                       // Cal O/P of hidden layer 2
        float in = 0;
        for (j = 0; j < NumHN1; j++)
          in += w2[j][i] * h1[j];
        h2[i] = (float) (1.0 / (1.0 + exp(double(-in))));  // Sigmoid
      }
      for (i = 0; i < NumOPs; i++) {                       // Cal O/P of output layer
        float in = 0;
        for (j = 0; j < NumHN2; j++) {
          in += w3[j][i] * h2[j];
        }
        y[i] = (float) (1.0 / (1.0 + exp(double(-in))));   // Sigmoid
      }
      // Cal error for this pattern
      PatErr = 0.0;
      for (i = 0; i < NumOPs; i++) {
        float err = y[i] - d[p][i]; // actual-desired O/P
        if (err > 0) PatErr += err; else PatErr -= err;
		//added for binary classification problem
        NumErr += ((y[i] < 0.5 && d[p][i] >= 0.5) || (y[i] >= 0.5 && d[p][i] < 0.5)); 
      }
      if (PatErr < MinErr) MinErr = PatErr;
      if (PatErr > MaxErr) MaxErr = PatErr;
      AveErr += PatErr;
      delete h1;
      delete h2;
      delete y;
    } else if (NumHN == 3) {
      float *h1 = new float[NumHN1];                       // O/Ps of hidden layer 1
      float *h2 = new float[NumHN2];                       // O/Ps of hidden layer 2
      float *h3 = new float[NumHN3];                       // O/Ps of hidden layer 3
      float *y = new float[NumOPs];                        // O/P of Net
      for (i = 0; i < NumHN1; i++) {                       // Cal O/P of hidden layer 1
        float in = 0;
        for (j = 0; j < NumIPs; j++)
          in += w1[j][i] * x[p][j];
        h1[i] = (float) (1.0 / (1.0 + exp(double(-in))));  // Sigmoid
      }
      for (i = 0; i < NumHN2; i++) {                       // Cal O/P of hidden layer 2
        float in = 0;
        for (j = 0; j < NumHN1; j++)
          in += w2[j][i] * h1[j];
        h2[i] = (float) (1.0 / (1.0 + exp(double(-in))));  // Sigmoid
      }
      for (i = 0; i < NumHN3; i++) {                       // Cal O/P of hidden layer 3
        float in = 0;
        for (j = 0; j < NumHN2; j++)
          in += w3[j][i] * h2[j];
        h3[i] = (float) (1.0 / (1.0 + exp(double(-in))));  // Sigmoid
      }
      for (i = 0; i < NumOPs; i++) {                       // Cal O/P of output layer
        float in = 0;
        for (j = 0; j < NumHN3; j++) {
          in += w4[j][i] * h3[j];
        }
        y[i] = (float) (1.0 / (1.0 + exp(double(-in))));   // Sigmoid
      }
      // Cal error for this pattern
      PatErr = 0.0;
      for (i = 0; i < NumOPs; i++) {
        float err = y[i] - d[p][i]; // actual-desired O/P
        if (err > 0) PatErr += err; else PatErr -= err;
		//added for binary classification problem
        NumErr += ((y[i] < 0.5 && d[p][i] >= 0.5) || (y[i] >= 0.5 && d[p][i] < 0.5)); 
      }
      if (PatErr < MinErr) MinErr = PatErr;
      if (PatErr > MaxErr) MaxErr = PatErr;
      AveErr += PatErr;
      delete h1;
      delete h2;
      delete h3;
      delete y;
    }
  }
  AveErr /= NumPats;
  float PcntErr = NumErr / float(NumPats) * 100.0;
  cout.setf(ios::fixed | ios::showpoint);
  cout << "Testing mlp:" << endl;
  cout << setprecision(6) << setw(7) << " " << setw(12) << "MinErr" << setw(12) << "AveErr" 
      << setw(12) << "MaxErr" << setw(12) << "\%PcntErr" << endl;
  cout << setprecision(6) << setw(8) << " " << setw(12) << MinErr << setw(12) << AveErr 
      << setw(12) << MaxErr << setw(12) << PcntErr << endl;

  if (NumHN == 1 || NumHN == 2 || NumHN == 3) {
    Free2DAry(w1, NumIPs);
    Free2DAry(w11, NumIPs);
    Free2DAry(w111, NumIPs);
    Free2DAry(w2, NumHN1);
    Free2DAry(w22, NumHN1);
    Free2DAry(w22, NumHN1);
  }
  if (NumHN == 2 || NumHN == 3) {
    Free2DAry(w3, NumHN2);
    Free2DAry(w33, NumHN2);
    Free2DAry(w333, NumHN2);
  }
  if (NumHN == 3) {
    Free2DAry(w4, NumHN3);
    Free2DAry(w44, NumHN3);
    Free2DAry(w444, NumHN3);
  }
}

float **Aloc2DAry(int m, int n) {
//Allocates memory for 2D array
  float **Ary2D = new float*[m];
  if (Ary2D == NULL) {
    cout << "No memory!\n";
    exit(1);
  }
  for (int i = 0; i < m; i++) {
    Ary2D[i] = new float[n];
    if (Ary2D[i] == NULL) {
      cout << "No memory!\n";
      exit(1);
    }
  }
  return Ary2D;
}

void Free2DAry(float **Ary2D, int n) {
//Frees memory in 2D array
  for (int i = 0; i < n; i++)
    delete[] Ary2D[i];
  delete[] Ary2D;
}

/*
 * implementation for selecting N patterns randomly from n patterns
 */
void RandomPattern(vector<int> &x) {
  // randomly select N items from n items
  int n = x.size();
  if (n > 1) {
    srand(time(NULL));
    for (int i = 0; i < n; ++i) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      int t = x[j];
      x[j] = x[i];
      x[i] = t;
    }
  }
}

/*
 * implementation for swaping two patterns randomly
 */
void RandomSwap(vector<int> &x) {
  int n = x.size();
  if (n > 1) {
    srand(time(NULL));
    int idx1 = rand() % n;
    int idx2 = rand() % n;
    int t = x[idx1];
    x[idx1] = x[idx2];
    x[idx2] = t;
  }
}

/*
 * implementation for getting mean & deviation
 * http://www.d.umn.edu/~deoka001/Normalization.html
 */
void GetMD(float **x, vector<float> &m, vector<float> &d, int NumPats){
  int NumX = m.size();
  for(int i=0; i<NumPats; ++i){
    for(int j=0; j<NumX; ++j){
      m[j]+=x[i][j];
    }
  }
  // mean
  for(int j=0; j<NumX; ++j){
    m[j]/=NumPats*1.0;
  }
  //deviation
  for(int i=0; i<NumPats; ++i){
    for(int j=0; j<NumX; ++j){
      d[j]+=pow((x[i][j]-m[j]), 2);
    }
  }
  for(int j=0; j<NumX; ++j){
    d[j]= sqrt(d[j]/NumPats*1.0);
  }
}

/*
 * implementation for normalizing data
 */
void Normalize(float **x, vector<float> &m, vector<float> &d, int NumPats) {
  int NumX = m.size();
  for(int i=0; i<NumPats; ++i){
    for(int j=0; j<NumX; ++j){
      x[i][j]= (x[i][j]-m[j])/d[j];
    }
  }
}
