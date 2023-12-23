// ANSI C-code for simulations accompanying
// "Role of efficient neurotransmitter release in barrel map development"
// Lu HC, Butts DA, Kaeser PS, She WC, Janz R, Crair MC
// Journal of Neuroscience Vol. 26, pp. 2692-2703 (2006)

// Code written by DA Butts to be compiled on UNIX systems
// To compile:  c++ BRLSmodel.c -O3 -lm -o sim
// To run:      sim 32000 0.3 0.88 1 P03seg1.dat
//   where the example input parameters here correspond to:
//     Simulation time (sec): 32000
//     Prob of release (Pr):  0.3
//     Cortical threshold:    0.88
//     Random number seed:    1
//     Data filename:         P03seg1.dat

// Use a negative threshold to activate "training" - which turns off synaptic
// weight adjustment and data-taking.  This is used to determine what the initial
// threshold of the simulation should be.  Simulation will use positive version of
// threshold and integrate firing rate over 10 times longer.

// Terminology note:
// Being from a visual-cortex-biased perspective, I often refer to
// the two sources of thalamic input considered in this simulation as
// left and right eyes - as opposed to primary and adjacent whiskers

//#define PHYS_TEST  0    // Display: 0 = left, 1 = right

// Uncomment 'DISPLAY' to display simulation events as the occur
//#define DISPLAY 1
//#define DETAILED_STATUS

// Simulation Output parameters
#define T_BEFORE_REPORT 0 //30000
#define REPORT_INTERVAL 100.0  //sec
#define TRAIN_TIME 4000.0 // sec

// Simulation speed - speed up long-evolving sims
//#define FASTSIM

// Synaptic facilitation (not used in current study)
#define FACIL 1

// Simulation time step
#define DT 0.001   // sec

// Number of Left and Right Whiskers
// (assume L is primary whisker PW, R is adjacent AW, but could be symmetric)
#define NL 20
#define NR 10

// Number of independent release sites per synapse
#define NVESICLES 10

// TC synapse properties
#define SPON_RATE  8.0   // spks/sec  (8 spks/sec average reported)
#define BURST_RATE 36.0  // spks/sec  (36 spks/sec average reported)

// Burst properties
#define BURST_FREQ 0.05  // Hz
#define BURST_TIME 0.10  // sec
#define BURST_INVOLVEMENT 1.0
#define JITTER 0.005  // sec
#define POISSON_BURST    // if selected, jitter only applies to first spike

// Postsynaptic cell
#define MEM_TAU  0.1 // sec

// Learning rule parameters
#ifdef FASTSIM
  #define LEARNING_RATE 0.0004 
#else
 #define LEARNING_RATE 0.0002 
#endif
#define TAU_POTENTIATION 0.01  // sec
#define TAU_DEPRESSION 0.05   // sec

/**** Modify this paramtere to change balance between potentiation and depression *****/
#define A_DEPRESSION  0.52  // (normal 0.52) - relative to potentiating window ( = 1) 


// EPSC Parameters
#define INH_DELAY  0.01   // sec
#define INH_MULT 0.5

#define EPSC_L 0.01   // sec
#define IPSC_L 0.05   // sec
#define EPSC_RP (BURST_TIME) 

// Initial spread in synaptic weights
#define DW 0.01

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define MAX_RUNS 128
#define ulong  unsigned long
#define uint  unsigned int
#define DECAY_MULT exp(-DT/MEM_TAU)
#define TLTP ((int)(TAU_POTENTIATION/DT))
#define TLTD ((int)(TAU_DEPRESSION/DT))
#define LLTP ((int)(7*TLTP))
#define LLTD ((int)(7*TLTD))
#define LEPSC ((int)(EPSC_L/DT))
#define EPSC_L_MOD (EPSC_L-DT/2)  // mod to make sure not bothered by integer time steps
#define LIPSC ((int)(IPSC_L/DT))
#define pBURST (DT*BURST_FREQ)
#define pSPKBURST (DT*BURST_RATE)
#define pSPKSPON  (DT*SPON_RATE)
#define ISI_BURST (1.0/BURST_RATE)
#define RDT (1.0/DT)
// Calc exh and inh so total integrates to 1
//#define MAG_EXC (1.0/ (((double)LEPSC)*exp(-EPSC_L/MEM_TAU)))
//#define MAG_INH (1.0/(((double)LIPSC)*exp(-IPSC_L/MEM_TAU)))

// Defines for Random number generator
#define M 714025  // max integer that may rightfully be requested
#define A 1366
#define C 150889

struct RandGen 
  {
  private:
    int dum, y, r[98];
    double w, z, x1,x2;
  public:
    RandGen( int = 1 );
    int Integer( int );
    double Real( void );
    double Gaussian( void );
    void init( int );
  };


struct synapse
  {
  int number, bursting, spiking;
  int i, t, n, inh_index, spon_spk;
  double a, cur;
  double w[NVESICLES], pf[NVESICLES];
  double w_inh;
  double last_epsc[NVESICLES];
  double burst_start, spike_time, next_spike, next_inh[10];
  double latent_reward[NVESICLES];
  
  synapse( int n, double initial_w );
  void burst( void );
  double current( void );
  double reward( void );
  double tot_strength( void );
  };

// Global variables
RandGen *rgen;
double master, last_post_spk;
double LTP[LLTP], LTD[LLTD], epsc_shape[LEPSC], ipsc_shape[LIPSC];
double p, w_cap;
double Lchange, Rchange;
int training, DGN;

/*********************************
               MAIN
*********************************/
int main( int argv, char *argc[] )
{
FILE *segfile;
double T, input, wnorm;
//double Lcor, Rcor, Rgauss, Rmult;
//double rateL, rateR;
double next_report, nspks_intvl;
double a, b, Lav = 0, Rav = 0;
uint Nrepeats, repeat;
double avrate, Lseg[MAX_RUNS], Rseg[MAX_RUNS];
double entropy, aventropy, avlattoLburst, avlattoRburst, NRbs, NLbs;
int pspk_type, pspk_type_last;
uint i, j, NLbursts, NRbursts, NLburstSpks, NRburstSpks;
ulong seed;
synapse *Leye[NL], *Reye[NR];
double Nspks, Lnet, Rnet, Laltpun, Raltpun;
double NS, NLB, NRB, Rcumburst, Lcumburst, Rcumspont, Lcumspont;
double last_Lburst, last_Rburst;
double V, theta, Lbias;
double RI, summer, summer2;

if (argv < 4)
  { printf( "\nUsage: %s [sim time (sec)] [p] [threshold] <seed> <seg filename>\n\n", argc[0] );  return 1; }

sscanf( argc[1], "%lf", &T );
sscanf( argc[2], "%lf", &p );
sscanf( argc[3], "%lf", &theta );

if (argv > 4)  sscanf( argc[4], "%lu", &seed );
else seed = 0;  

// Negative threshold activates "training mode"
if (theta > 0)  { training = 0;  DGN = 0; RI = REPORT_INTERVAL;  }
else  { training = 1;  DGN = 1;  theta *= -1;  RI = 10*REPORT_INTERVAL;  T = TRAIN_TIME;  }
DGN = 0;

// Bias in weights towards left 
Lbias = 1;

// Define learning rule (pre-calculate to save time during sim)
for( i = 0; i < LLTP; i ++ )  LTP[i] = LEARNING_RATE * exp(-((double)i)/LLTP);
for( i = 0; i < LLTD; i ++ )  LTD[i] = LEARNING_RATE * A_DEPRESSION*exp(-((double)i)/LLTD);

// Define EPSC and IPSC shape - box epsc at the moment
for( i = 0; i < LEPSC; i ++ )  epsc_shape[i] =  1.0/((double)LEPSC);    //epsc_shape[i] =  MAG_EXC;
for( i = 0; i < LIPSC; i ++ )  ipsc_shape[i] = -1.0/((double)LIPSC);    //ipsc_shape[i] = -MAG_INH;

// Set cap on maximum strength of synapse
w_cap = 0.25*theta/(p*NVESICLES);  //0.5*theta/(p*NVESICLES);

// Open data file
if (training == 0)
  {
  if (argv > 5)  segfile = fopen( argc[5], "w" );
  else  segfile = fopen( "seg.dat", "w" );
  if (!segfile)  printf( "Couldn't open data file.  oh well.\n" );
  }
  
// Initialize simulation variables
avrate = 0;
Lnet = 0;  Rnet = 0;
NS = -1;  NLB = 0;  NRB = 0;
Rcumburst = 0;  Lcumburst = 0;  Raltpun = 0;
Rcumspont = 0;  Lcumspont = 0;  Laltpun = 0;


// WANT MULTIPLE REPEATS - put loop here

  // Make random number generator object
  rgen = new RandGen( seed ++ );

  // Set weights - normalized so sum total is 100
  wnorm = 0;
  for( i = 0; i < NL; i ++ )
    { Leye[i] = new synapse( i, (1+DW*rgen->Gaussian())*Lbias );  wnorm += Leye[i]->tot_strength(); }
  for( i = 0; i < NR; i ++ )
    { Reye[i] = new synapse( i+NL, (1+DW*rgen->Gaussian()) );  wnorm += Reye[i]->tot_strength(); }
  wnorm = 50.0/wnorm;
  for( i = 0; i < NL; i ++ )
    {
    for( j = 0; j < NVESICLES; j ++ )  (Leye[i]->w[j]) *= wnorm;
	  Leye[i]->w_inh *= wnorm;
    }
  for( i = 0; i < NR; i ++ )
    {
	  for( j = 0; j < NVESICLES; j ++ )  (Reye[i]->w[j]) *= wnorm;
	  Reye[i]->w_inh *= wnorm;
    }
  Nspks = 0;  NLburstSpks = 0;  NRburstSpks = 0;  NLbursts = 0;  NRbursts = 0;
  nspks_intvl = 0;  next_report = RI;  avlattoLburst = 0;  avlattoRburst = 0;
  NRbs = 0;  NLbs = 0;

  // Sim variable reset
  last_post_spk = -100;  last_Rburst = -100;  last_Lburst = -100;
  V = 0;

  /*******************************************/
  /****          SIMULATION LOOP          ****/
  /*******************************************/
  for( master = 0; master <= T; master += DT )
    {
    // Determine if a new burst occurs 
    if (((master-last_Lburst) > BURST_TIME) && (rgen->Real() < pBURST))
	    { 
      for( i = 0; i < NL; i ++ )  if (rgen->Real() < BURST_INVOLVEMENT)  Leye[i] -> burst();
	    NLbursts ++;
	    last_Lburst = master;
	    #ifdef DISPLAY
	      if (master > T_BEFORE_REPORT)  printf( "\nLeft burst:\t\t%0.3lf\n", master );
	    #endif
      }
    if (((master-last_Rburst) > BURST_TIME) && (rgen->Real() < pBURST))
	    {
	    for( i = 0; i < NR; i ++ )  if (rgen->Real() < BURST_INVOLVEMENT)  Reye[i] -> burst();
	    NRbursts ++;
	    last_Rburst = master;
	    }

	  // Determine membrane potential
    V *= DECAY_MULT;

    // Integrate current (i.e. voltage change - not biophysical model) from all synapses
	  a = 0;
	  for( i = 0; i < NL; i ++ )  a += Leye[i] -> current();

	  #ifdef PHYS_TEST
	    printf( " %7.3lf\tL %lf\t", master, a );
	  #endif
	  V += a;
	
	  a = 0;
	  for( i = 0; i < NR; i ++ )  a += Reye[i] -> current();

	  #ifdef PHYS_TEST
	    printf( " R %lf\t\t", a );
	  #endif
	  V += a;	  
	  
	  #ifdef PHYS_TEST
	    printf( "V = %lf\n", V );
    #endif

	  // Postsynaptic spike
    if (V > theta)
      {
      Nspks ++;  nspks_intvl ++;
	    last_post_spk = master;
	    pspk_type_last = pspk_type;
	    pspk_type = 0;
	    if ((master-last_Lburst) < BURST_TIME)
	      {
		    NLburstSpks ++;  NLbs ++;
		    avlattoLburst += (master-last_Lburst)*1000;
		    pspk_type = 1;
        }
	    if ((master-last_Rburst) < BURST_TIME)
		    { 
		    NRburstSpks ++;  NRbs ++;
		    avlattoRburst += (master-last_Rburst)*1000;
		    pspk_type = 2;
        }
	
	    // Add changes from depression following last post-spike
	    Lnet += Lchange;  Rnet += Rchange;
      if (pspk_type_last == 0)
	      {
	      NS ++;
		    Rcumspont += Rnet;
		    Lcumspont += Lnet;
		    }
	    if (pspk_type_last == 1)
		    {
		    NLB ++;
		    Lcumburst += Lnet;
	      Raltpun += Rnet;
		    }
	    if (pspk_type_last == 2)
	      {
		    NRB ++;
        Rcumburst += Rnet;
		    Laltpun += Lnet;
		    }

      if (DGN)
	      if (master > T_BEFORE_REPORT)
		      {
		      printf( "(%5.1lf / %5.1lf)\t", Lnet/LEARNING_RATE, Rnet/LEARNING_RATE );
		      if (pspk_type_last == 0)  printf( " Tot: %5.1lf / %5.1lf\n", Lcumspont/(LEARNING_RATE*NS), Rcumspont/(LEARNING_RATE*NS) );
		      if (pspk_type_last == 1)  printf( " Tot: %5.1lf\n", Lcumburst/(LEARNING_RATE*NLB) );
		      if (pspk_type_last == 2)  printf( " Tot:         %5.1lf\n", Rcumburst/(LEARNING_RATE*NRB) );
		      printf( "%0.0lf  %0.3lf\t", NS, master );
		      if (pspk_type == 0)  printf( "S Spk  XXX  " );
		      if (pspk_type == 1)  printf( "L Spk %4.0lf  ", 1000*(master-last_Lburst) );
		      if (pspk_type == 2)  printf( "R Spk %4.0lf  ", 1000*(master-last_Rburst) );
		      }

	    Lchange = 0;  Rchange = 0;
      // Add potentiation from preceding this spike
      for( i = 0; i < NL; i ++ )  Lchange += Leye[i] -> reward();
      for( i = 0; i < NR; i ++ )  Rchange += Reye[i] -> reward();

	    Lnet = Lchange;  Rnet = Rchange;
      Lchange = 0;  Rchange = 0;
	    // Reset Voltage
	    V = 0;
	    }

	if (master >= next_report)
	  {
	  if (training == 0)  fprintf( segfile, "%7.3lf  ", nspks_intvl/RI );
	  next_report = master + RI;

	  if (training == 0)
	    {
		  Lav = 0;  Rav = 0;
	    for( i = 0; i < NL; i ++ )
	      {
		    Lav += Leye[i]->tot_strength();
	      //Lseg[repeat] += Leye[i]->w;
	      fprintf( segfile, "%6.3lf ", Leye[i]->tot_strength() );
        //if (Leye[i]->w > 0)  entropy += 0.01*(Leye[i]->w)*log(0.01*(Leye[i]->w));
	      }
	    for( i = 0; i < NR; i ++ )
	      {
		    Rav += Reye[i]->tot_strength();
	      //Rseg[repeat] += Reye[i]->w;
	      fprintf( segfile, "%6.3lf ", Reye[i]->tot_strength() );
        //if (Reye[i]->w > 0)  entropy += 0.01*(Reye[i]->w)*log(0.01*(Reye[i]->w));
	      }
	    fprintf( segfile, "\n" );
		  }
    
    summer = 0;  for( i = 0; i < NL; i ++ )  summer += Leye[i]->tot_strength();
    summer2 = 0;  for( i = 0; i < NR; i ++ )  summer2 += Reye[i]->tot_strength();
    
    printf( "%5.0lf  %5.2lf Hz\t\tL%5.2lf  R%5.2lf\n", master, nspks_intvl/RI, summer, summer2 );
    
	  #ifdef DETAILED_STATUS
	    //printf( "(%3.0lf / %3.0lf ms lat): ", avlattoLburst/NLbs, avlattoRburst/NRbs );
	    //printf( "\t%0.2lf\t%0.2lf\n", Lav/NL, Rav/NR ); fflush(stdout);

      printf( "B: %0.1lf / %0.1lf (x%0.0lf / x%0.0lf)  ", Lcumburst/(NLB*LEARNING_RATE), Rcumburst/(NRB*LEARNING_RATE), NLB, NRB );
      printf( "S: %0.1lf / %0.1lf (x%0.0lf)  ", Lcumspont/(NS*LEARNING_RATE), Rcumspont/(NS*LEARNING_RATE), NS );
	    printf( "C: %0.1lf / %0.1lf (x%0.0lf/x%0.0lf)\n", Laltpun/(NRB*LEARNING_RATE), Raltpun/(NLB*LEARNING_RATE), NRB, NLB );
    #endif
    Lcumburst = 0;  Rcumburst = 0;  NLB = 0;  NRB = 0;  NS = 0;  Laltpun = 0;  Raltpun = 0;  Lcumspont = 0;  Rcumspont = 0;
	  nspks_intvl = 0;  avlattoLburst = 0;  avlattoRburst = 0;  NLbs = 0;  NRbs = 0;
	  }
  }
  
avrate += Nspks/((double)T);

if (training)
  {
  printf( "Average postsynaptic firing rate (%0.0lf): %0.2lf Hz\n", Nspks, Nspks/((double)T) );
  printf( "%d left bursts (%d spikes), %d right bursts (%d spikes)\n\n", NLbursts, NLburstSpks, NRbursts, NRburstSpks );
  printf( "B: %5.1lf / %5.1lf (x%0.0lf / x%0.0lf)\n", Lcumburst/(NLB*LEARNING_RATE), Rcumburst/(NRB*LEARNING_RATE), NLB, NRB );
  printf( "S: %5.1lf / %5.1lf (x%0.0lf)\n", Lcumspont/(NS*LEARNING_RATE), Rcumspont/(NS*LEARNING_RATE), NS );
  printf( "C: %5.1lf / %5.1lf (x%0.0lf/x%0.0lf)\n", Laltpun/(NRB*LEARNING_RATE), Raltpun/(NLB*LEARNING_RATE), NRB, NLB );
	}

if (training == 0)   fclose( segfile );
  
// Deallocate memory
for( i = 0; i < NL; i ++ )  delete( Leye[i] );
for( i = 0; i < NR; i ++ )  delete( Reye[i] );
delete(rgen);

return 0;
}


synapse::synapse( int n, double initial_w )
  {
  number = n;
  bursting = 0;  spiking = 0;
  for( i = 0; i < NVESICLES; i ++ ) 
    { w[i] = initial_w;  latent_reward[i] = 0; last_epsc[i] = -1000;  pf[i] = 1; }
  w_inh = (initial_w*NVESICLES*p) * INH_MULT;
  next_spike = -EPSC_L;  for( i = 0; i < 10; i ++ )  next_inh[i] = -IPSC_L;
  inh_index = 0;
  }

void synapse::burst( void )
  {
  burst_start = master;
  bursting = 1;
  #ifdef JITTER
    next_spike = master + JITTER*rgen->Real();    
  #endif
  }

double synapse::current( void )
// Calculates synaptic current at the current time step from the particular synapse
  {
  cur = 0;
  // Check for inhibitory inputs first
  if (master < (next_inh[inh_index]+IPSC_L))  // then still inhibition to resolve
    {
	  for( i = 0; i < 10; i ++ )
      if ((master < (next_inh[i]+IPSC_L)) && (master >= next_inh[i]))
        {
	      t = (int)((master-next_inh[i])*RDT);
        cur += w_inh*ipsc_shape[t];
	      }
	  }
  if (spiking)
    {  // note that can only be firing one epsc at a time
	  if ((master-spike_time) < EPSC_L_MOD)
	    {
	    for( i = 0; i < NVESICLES; i ++ )
	      {
	      t = (int)((master-last_epsc[i])*RDT);
	      if (t < LEPSC)  cur += w[i]*epsc_shape[t];  // only contributes to recently-released vesicles
		    }

	    return( cur );
	    }
	  else spiking = 0;
	  }

  if (bursting)
    {
	  if ((master - burst_start) >= BURST_TIME)
	    {
	    for( i = 0; i < NVESICLES; i ++ )  pf[i] = 1;
	    bursting = 0;
	    }
	  else
	    {
	    #ifdef POISSON_BURST
	      if (next_spike > 0) // then first burst-spike scheduled
          {
		      if (next_spike > master)  return( cur );
          }
		    else  if (rgen->Real() >= pSPKBURST)  return( cur );
	    #else
	      if (master < next_spike)  return( cur );
	    #endif
	    //printf( "  BurstSpk %lf ", master );
      spon_spk = 0;
	    }
	  }
  else 
    {
	  if (rgen->Real() >= pSPKSPON)  return( cur );
    spon_spk = 1;
    //printf( " SpSpk " );
	  }

  // If still here, then firing spike (spontaneous or part of burst)
  spiking = 1;
  spike_time = master;

  // Delayed inhibition (for burst spikes)
  inh_index ++;  if (inh_index >= 10)  inh_index = 0;
  next_inh[inh_index] = master + INH_DELAY;
    
  #ifdef PHYS_TEST
    if (bursting)   printf( "\n    Burst %2d spk ", number );
  #endif
  
  #ifdef POISSON_BURST
    next_spike = 0;
  #else
    if (bursting)  next_spike = master + ISI_BURST + JITTER*(rgen->Real()-0.5);
  #endif
  
  // Update any latent rewards
  if (training < 1)  
    for( i = 0; i < NVESICLES; i ++ )
	    {
      w[i] += latent_reward[i];
      if (w[i] < 0)  w[i] = 0;
      if (w[i] > w_cap)  w[i] = w_cap;
	    #ifdef DISPLAY
        if ((number == DISPLAY) && (latent_reward[i] != 0) && (master > T_BEFORE_REPORT))  
	  	    printf ( "    Update: %d: %lf (now %lf)\n", i, latent_reward[i], w[i] );
      #endif
	    latent_reward[i] = 0;
      }

  #ifdef DISPLAY
    if ((number == DISPLAY) && (master > T_BEFORE_REPORT)) printf( "%0.3lf\tSpike ", master );
  #endif

  // Is a vesicle released?
  a = 0;
  for( i = 0; i < NVESICLES; i ++ )
    if (((master-last_epsc[i]) > EPSC_RP) && (rgen->Real() < p*pf[i]))  // can put in facilitation term that temp increases p here
	    {
      #ifdef PHYS_TEST
	    if (bursting)
        printf( "v%d", i );
      #endif
	    #ifdef DISPLAY
        if ((number == DISPLAY) && (master > T_BEFORE_REPORT))  printf( "v%d", i );
	    #endif
	    // Make EPSC
      last_epsc[i] = master;  
      cur += w[i]*epsc_shape[0];  //printf( "\n\t  xx %d  %lf", i, cur );
 	    a += w[i];
      // Punish synapse (spike follows post-spike)
      t = (int)((master-last_post_spk)*RDT);
      if (t < LLTD)  
        {
        latent_reward[i] += -LTD[t]; 
        if (number < NL)  Lchange += -LTD[t];
        else  Rchange += -LTD[t];
        //printf( "  %0.3lf  Punish %d\n", master, t ); 
        }
      }
    else if (((master-last_epsc[i]) > EPSC_RP) && bursting) // then facilitate
      pf[i] *= FACIL;  //printf( "facil %lf\n", pf[i] ); }

  #ifdef PHYS_TEST
    if (bursting)  printf( "\t\t%lf\n\t", a );
  #endif
	  
  #ifdef DISPLAY
    if ((number == DISPLAY) && (master > T_BEFORE_REPORT))  printf( "\tCur = %lf\n", cur );  
  #endif

  return( cur );
  }

double synapse::reward( void )
  {
  cur = 0;  n = 0;
  for( i = 0; i < NVESICLES; i ++ )
    {
    t = (int)((master-last_epsc[i])*RDT);
    if (t < LLTP)
	    {
      latent_reward[i] += LTP[t];
	    cur += LTP[t];  n ++;
	    }
    //if (number == DISPLAY)  printf( "+(%d %lf)", i, LTP[t] ); }
	  }
  return( cur );
  } 

double synapse::tot_strength( void )
  {
  cur = 0;
  for( i = 0; i < NVESICLES; i ++ )  cur += w[i];
  return( 10*cur );
  }

// Random-number generator class
RandGen::RandGen( int seed )
  { init(seed); }
	
void RandGen::init( int seed )
  {
  int j;
  seed ++;
  dum = (-seed);

  if ((dum = (C-dum) % M) < 0)  dum= -dum;
  for( j = 1; j <= 97; j++) 
    {
    dum = ((A*dum)+C) % M;
    r[j] = dum;
    }
	
  dum = (A*dum+C) % M;
  y = dum;
  }

int RandGen::Integer( int max )
  {
  int j;

  j = 1 + (int)(97.0*y/(double)M);
  if (j > 97 || j < 1) 
    {
    printf("Error in random number generator!\n");
    exit( 1 );
    }
  y = r[j];
  dum = (A * dum + C) % M;
  r[j] = dum;

  return( y % max );
  }

double RandGen::Real( void )
  {
  int j;

  j = 1 + (int)(97.0*y/(double)M);
  if (j > 97 || j < 1) 
    {
    printf("Error in random number generator!\n");
    exit(1);
    }
  y = r[j];
  dum = (A * dum + C) % M;
  r[j] = dum;

  return( (double)y/(double)M );
  }

double RandGen::Gaussian( void )
  {
  do 
    {
    x1 = 2.0 * Real() - 1.0;
    x2 = 2.0 * Real() - 1.0;
    w = x1*x1 + x2*x2;
    }
  while (w >= 1.0  || w == 0.0);
  z = sqrt( (-2.0*log(w)) / w );
  return( x2*z);
  }
