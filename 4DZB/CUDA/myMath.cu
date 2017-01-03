#include "myMath.cuh"



int __inline iDiviUp(int a, int b)
{
	return a %b ? a / b + 1 : a / b;
}