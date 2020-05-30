#include "genParticlesFunctions.h"

void createParticlesOilSpill2D(double DL, double mult, const char filename[]) {

	int mp = mult * 200;
	int np = mult * 25;

	double xl = mp * DL;
	double yl = mp * DL;

	double dx = xl / mp;
	double dy = yl / mp;

	double vx = 0.0;
	double mp2 = (double)mp;

	double offsetXwall = xl * 2 - 28 * dx * mult;
	int sizeXWater = mp2 * 465 * dx;

	FILE* writeGrid;
	fopen_s(&writeGrid,filename, "w");
	//fprintf(writeGrid, "0.0\n");
	fprintf(writeGrid, "%d \n", ((sizeXWater - 1) * np) + (((28 * mult) - 2) * (2 * np + 13)) /**/ + (5 * mp / 2 + 48) /**/ + (5 * mp / 2 + 52) /**/ + (5 * mp / 2 + 56) /*middle wall ->*/ + (mp / 20) * 3 + 3 * ((mp / (10)) * 3 - (20 * (mult - 1))));

	//PARTICULAS TIPO -1 
	//for (int i = 0; i < mp + 5; ++i){ //up
	//	fprintf(writeGrid,  ""3 " << i*dx + dx << " " << yl + (5*dy) << " " << vx << " 0.0 0.0 0.0" << endl;
	//}
	for (int i = 0; i < (2 * mp) + 5; ++i) { //down
		fprintf(writeGrid, "%lf 0 -1 0 0 0 \n", (i * dx));
	}
	for (int i = 0; i < (mp) / 4 + 25; ++i) { //left
		fprintf(writeGrid, "0 %lf -1 0 0 0 \n", ((i * dy) + dy));
	}
	for (int i = 0; i < (mp) / 4 + 26; ++i) { //right
		fprintf(writeGrid, "%lf %lf -1 0 0 0 \n", (2 * xl + (5 * dx)), (i * dy));
	}

	//PARTICULAS TIPO 0 #externa
	//for (int i = 0; i < (mp) + 3; ++i){ //up
	//  fprintf(writeGrid, "3  %lf %lf %lf 0.0 0.0 0.0 \n", ((i*dx) + (2*dx)), (yl + (4*dy)), vx);
	//}
	for (int i = 0; i < (2 * mp) + 3; ++i) { //down
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", ((i * dx) + dx), dy);
	}
	for (int i = 0; i < (mp) / 4 + 24; ++i) { //left
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", dx, ((i * dy) + (2 * dy)));
	}
	for (int i = 0; i < (mp) / 4 + 25; ++i) { //right
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", (2 * xl + (4 * dx)), ((i * dy) + dy));
	}

	//PARTICULAS TIPO 0 #interna
	//for (int i = 0; i < (mp) + 1; ++i){ //up
	//	fprintf(writeGrid, "2  %lf %lf %lf 0.0 0.0 0.0 \n", ((i*dx) + 3*dx), (yl + 3*dy), vx);
	//}
	for (int i = 0; i < (2 * mp) + 1; ++i) { //down
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", ((i * dx) + 2 * dx), (2 * dy));
	}
	for (int i = 0; i < (mp) / 4 + 23; ++i) { //left
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", (2 * dx), ((i * dy) + (3 * dy)));
	}
	for (int i = 0; i < (mp) / 4 + 24; ++i) { //right
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", (2 * xl + 3 * dx), ((i * dy) + (2 * dy)));
	}

	//PARTICULAS TIPO 1 #fluido
	//for (int i = 0; i < (mp / 2); i++) {
	//	for (int j = 0; j < np/2; j++) {
	//		fprintf(writeGrid, "%lf %lf 2 0 0 0 \n", (i*dx + (3 * dx)), (j*dy + (3 * dy)));
	//	}
	//}

	//WALL
	for (int i = 0; i < 3; ++i) { //down
		for (int j = 0; j < mp / 20; j++) {
			fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", i * dx + (2 * dx) + offsetXwall, ((j * dy) + (3 * dy)));
		}
	}

	for (int i = 0; i < 3; ++i) { //up
		for (int j = (mp / 20 + 3); j < (mp / 20 + 3) + ((mp / (10)) * 3) - 20 * (mult - 1); ++j) {
			fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", i * dx + (2 * dx) + offsetXwall, ((j * dy) + (3 * dy)));
		}
	}


	for (int i = 0; i < sizeXWater - 1; i++) {
		for (int j = 0; j < np; j++) {
			fprintf(writeGrid, "%lf %lf 1 0 0 0 \n", (i * dx + (3 * dx)), (j * dy + (3 * dy)));
		}
	}

	for (int i = sizeXWater + 2; i < sizeXWater + (28 * mult); i++) {
		for (int j = 0; j < 2 * np + 13; j++) {
			fprintf(writeGrid, "%lf %lf 2 0 0 0 \n", (i * dx + (3 * dx)), (j * dy + (3 * dy)));
		}
	}

	//PARTICULAS TIPO -1 #externa
	////for (int i = 0; i < mp + 5; ++i){ //up
	////	fprintf(writeGrid,  ""3 " << i*dx + dx << " " << yl + (5*dy) << " " << vx << " 0.0 0.0 0.0" << endl;
	////}
	//for (int i = 0; i < (2 * mp) + 7; ++i){ //down
	//	fprintf(writeGrid, "3 %lf %lf %lf 0.0 0.0 0.0 \n", (i*dx), -dy, vx);
	//}
	//for (int i = 0; i < (mp)+8; ++i){ //left
	//	fprintf(writeGrid, "3 %lf %lf 0.0 0.0 0.0 0.0 \n", -dx, (i*dy) - dy);
	//}
	//for (int i = 0; i < (mp)+7; ++i){ //right
	//	fprintf(writeGrid, "3 %lf %lf 0.0 0.0 0.0 0.0 \n", (2 * xl + (6 * dx)), (i*dy));
	//}

	//PARTICULAS DO TOCO DO TIPO 2
	/*for (int i = 0; i < 3; ++i){
	for (int j = 0; j < 8; j++){
	fprintf(writeGrid, "2 %lf %lf 0.0 0.0 0.0 0.0 \n", ((i*dx) + (mp + 1) * dx), (3 * dy) + j*dy);
	}
	}*/

	fclose(writeGrid);
}

void createParticlesDamBreak2D(double DL, double mult, const char filename[]) {

	int mp = 36*mult;
	int np = 36*mult;

	double xl = mp * DL;
	double yl = np * DL;

	double dx = xl / mp;
	double dy = yl / np;

	double vx = 0.0;

	FILE* writeGrid;
	fopen_s(&writeGrid,filename, "w");
	//fprintf(writeGrid, "0.0\n");
	fprintf(writeGrid, "%d \n", ((mp / 2) * np) + (4 * mp + 3 * 1 + 7) + (4 * mp + 3 * 3 + 5) + (4 * mp + 3 * 5 + 3) /*+ (4 * mp + 7 + 8 + 7) + 3 * 8*/);
	printf("%d \n", ((mp / 2) * np) + (4 * mp + 3 * 1 + 7) + (4 * mp + 3 * 3 + 5) + (4 * mp + 3 * 5 + 3) /*+ (4 * mp + 7 + 8 + 7) + 3 * 8*/);
	//PARTICULAS TIPO -1 
	//for (int i = 0; i < mp + 5; ++i){ //up
	//	fprintf(writeGrid,  ""3 " << i*dx + dx << " " << yl + (5*dy) << " " << vx << " 0.0 0.0 0.0" << endl;
	//}
	for (int i = 0; i < (2 * mp) + 5; ++i) { //down
		fprintf(writeGrid, "%lf 0 -1 0 0 0 \n", (i * dx));
	}
	for (int i = 0; i < (mp)+6; ++i) { //left
		fprintf(writeGrid, "0 %lf -1 0 0 0 \n", ((i * dy) + dy));
	}
	for (int i = 0; i < (mp)+7; ++i) { //right
		fprintf(writeGrid, "%lf %lf -1 0 0 0 \n", (2 * xl + (5 * dx)), (i * dy));
	}

	//PARTICULAS TIPO 0 #externa
	//for (int i = 0; i < (mp) + 3; ++i){ //up
	//  fprintf(writeGrid, "3  %lf %lf %lf 0.0 0.0 0.0 \n", ((i*dx) + (2*dx)), (yl + (4*dy)), vx);
	//}
	for (int i = 0; i < (2 * mp) + 3; ++i) { //down
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", ((i * dx) + dx), dy);
	}
	for (int i = 0; i < (mp)+5; ++i) { //left
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", dx, ((i * dy) + (2 * dy)));
	}
	for (int i = 0; i < (mp)+6; ++i) { //right
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", (2 * xl + (4 * dx)), ((i * dy) + dy));
	}

	//PARTICULAS TIPO 0 #interna
	//for (int i = 0; i < (mp) + 1; ++i){ //up
	//	fprintf(writeGrid, "2  %lf %lf %lf 0.0 0.0 0.0 \n", ((i*dx) + 3*dx), (yl + 3*dy), vx);
	//}
	for (int i = 0; i < (2 * mp) + 1; ++i) { //down
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", ((i * dx) + 2 * dx), (2 * dy));
	}
	for (int i = 0; i < (mp)+4; ++i) { //left
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", (2 * dx), ((i * dy) + (3 * dy)));
	}
	for (int i = 0; i < (mp)+5; ++i) { //right
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", (2 * xl + 3 * dx), ((i * dy) + (2 * dy)));
	}

	//PARTICULAS TIPO 1 #fluido
	//for (int i = 0; i < (mp / 2); i++) {
	//	for (int j = 0; j < np/2; j++) {
	//		fprintf(writeGrid, "%lf %lf 2 0 0 0 \n", (i*dx + (3 * dx)), (j*dy + (3 * dy)));
	//	}
	//}

	for (int i = 0; i < (mp / 2); i++) {
		for (int j = 0; j < np; j++) {
			fprintf(writeGrid, "%lf %lf 1 0 0 0 \n", (i * dx + (3 * dx)), (j * dy + (3 * dy)));
		}
	}

	//PARTICULAS TIPO -1 #externa
	////for (int i = 0; i < mp + 5; ++i){ //up
	////	fprintf(writeGrid,  ""3 " << i*dx + dx << " " << yl + (5*dy) << " " << vx << " 0.0 0.0 0.0" << endl;
	////}
	//for (int i = 0; i < (2 * mp) + 7; ++i){ //down
	//	fprintf(writeGrid, "3 %lf %lf %lf 0.0 0.0 0.0 \n", (i*dx), -dy, vx);
	//}
	//for (int i = 0; i < (mp)+8; ++i){ //left
	//	fprintf(writeGrid, "3 %lf %lf 0.0 0.0 0.0 0.0 \n", -dx, (i*dy) - dy);
	//}
	//for (int i = 0; i < (mp)+7; ++i){ //right
	//	fprintf(writeGrid, "3 %lf %lf 0.0 0.0 0.0 0.0 \n", (2 * xl + (6 * dx)), (i*dy));
	//}

	//PARTICULAS DO TOCO DO TIPO 2
	/*for (int i = 0; i < 3; ++i){
	for (int j = 0; j < 8; j++){
	fprintf(writeGrid, "2 %lf %lf 0.0 0.0 0.0 0.0 \n", ((i*dx) + (mp + 1) * dx), (3 * dy) + j*dy);
	}
	}*/

	fclose(writeGrid);
}

void createParticlesShearCavity2D(double DL, double mult, const char filename[]) {

	int mp = 36*mult;
	int np = 36*mult;

	double xl = mp * DL;
	double yl = np * DL;

	double dx = xl / mp;
	double dy = yl / np;

	FILE* writeGrid;
	fopen_s(&writeGrid,filename, "w");
	//fprintf(writeGrid, "0.0\n");
	fprintf(writeGrid, "%d \n", ((mp - 7) * (np - 6)) + /**/ 2 * (mp - 4) + 2 * (np - 4) + /**/ 2 * (mp - 2) + 2 * (np - 2) + /**/(4 * mp));

	//PARTICULAS TIPO -1 
	for (int i = 0; i < mp; ++i) { //down
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", (i * dx), 0.0);
	}
	for (int i = 0; i < np; ++i) { //left
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", (0.0 * dx), ((i * dy) + dy));
	}
	for (int i = 0; i < np; ++i) { //right
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", xl, (i * dy));
	}
	for (int i = 0; i < mp; ++i) { //up
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", (i * dx + dx), yl);
	}

	//PARTICULAS TIPO 0 #externa
	for (int i = 0; i < mp - 2; ++i) { //down
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", ((i * dx) + dx), dy);
	}
	for (int i = 0; i < np - 2; ++i) { //left
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", dx, ((i * dy) + (2 * dy)));
	}
	for (int i = 0; i < np - 2; ++i) { //right
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", (xl - dx), ((i * dy) + dy));
	}
	for (int i = 0; i < mp - 2; ++i) { //up
		fprintf(writeGrid, "%lf %lf 0 1.0 0 0 \n", ((i * dx) + 2 * dx), (yl - dy));
	}

	//PARTICULAS TIPO 0 #interna
	for (int i = 0; i < mp - 4; ++i) { //down
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", ((i * dx) + 2 * dx), 2 * dy);
	}
	for (int i = 0; i < np - 4; ++i) { //left
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", 2 * dx, ((i * dy) + (3 * dy)));
	}
	for (int i = 0; i < np - 4; ++i) { //right
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", (xl - 2 * dx), ((i * dy) + 2 * dy));
	}
	for (int i = 0; i < mp - 4; ++i) { //up
		fprintf(writeGrid, "%lf %lf 0 1.0 0 0 \n", ((i * dx) + 3 * dx), (yl - 2 * dy));
	}

	//PARTICULAS TIPO 1 #fluido
	for (int i = 0; i < (mp - 7); i++) {
		for (int j = 0; j < (np - 6); j++) {
			fprintf(writeGrid, "%lf %lf 1 0 0 0 \n", (i * dx + (4 * dx)), 4 * dy + dy * j);
		}
	}

	fclose(writeGrid);
}

void createParticlesWaterDrop2D(double DL, double mult, const char filename[]) {

	int mp = 18*mult;
	int np = 18*mult;
	int conta = 0;

	double xl = mp * DL;
	double yl = np * DL;

	double dx = xl / mp;
	double dy = yl / np;

	FILE* writeGrid;
	fopen_s(&writeGrid,filename, "w");
	//fprintf(writeGrid, "0.0\n");
	fprintf(writeGrid, "%d \n", 1005 /*conta variable value*/);


	//PARTICULAS TIPO 1 #fluido
	for (int i = -mp; i < mp; i++) {
		for (int j = -mp; j < mp; j++) {
			if (i * i + j * j < mp * mp) {
				fprintf(writeGrid, "%lf %lf 1 0 0 0 \n", i * dx + 2 * mp * dx, j * dx + 2 * mp * dx);
				conta++;
			}
		}
	}

	printf("contador: %d", conta);
	fclose(writeGrid);
}

void createParticlesRTinstability(double DL, double mult, const char filename[]) {

	int mp = 36*mult;
	int np = 36*mult;

	double xl = mp * DL;
	double yl = np * DL;

	double dx = xl / mp;
	double dy = yl / np;

	double vx = 0.0;

	FILE* writeGrid;
	fopen_s(&writeGrid,filename, "w");
	//fprintf(writeGrid, "0.0\n");
	fprintf(writeGrid, "%d \n", (2 * mp * np) + (4 * mp + 3 * 1 + 7) + (4 * mp + 3 * 3 + 5) + (4 * mp + 3 * 5 + 3) /*+ (4 * mp + 7 + 8 + 7) + 3 * 8*/);

	//PARTICULAS TIPO -1 
	//for (int i = 0; i < mp + 5; ++i){ //up
	//	fprintf(writeGrid,  ""3 " << i*dx + dx << " " << yl + (5*dy) << " " << vx << " 0.0 0.0 0.0" << endl;
	//}
	for (int i = 0; i < (2 * mp) + 5; ++i) { //down
		fprintf(writeGrid, "%lf 0 -1 0 0 0 \n", (i * dx));
	}
	for (int i = 0; i < (mp)+6; ++i) { //left
		fprintf(writeGrid, "0 %lf -1 0 0 0 \n", ((i * dy) + dy));
	}
	for (int i = 0; i < (mp)+7; ++i) { //right
		fprintf(writeGrid, "%lf %lf -1 0 0 0 \n", (2 * xl + (5 * dx)), (i * dy));
	}

	//PARTICULAS TIPO 0 #externa
	//for (int i = 0; i < (mp) + 3; ++i){ //up
	//  fprintf(writeGrid, "3  %lf %lf %lf 0.0 0.0 0.0 \n", ((i*dx) + (2*dx)), (yl + (4*dy)), vx);
	//}
	for (int i = 0; i < (2 * mp) + 3; ++i) { //down
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", ((i * dx) + dx), dy);
	}
	for (int i = 0; i < (mp)+5; ++i) { //left
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", dx, ((i * dy) + (2 * dy)));
	}
	for (int i = 0; i < (mp)+6; ++i) { //right
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", (2 * xl + (4 * dx)), ((i * dy) + dy));
	}

	//PARTICULAS TIPO 0 #interna
	//for (int i = 0; i < (mp) + 1; ++i){ //up
	//	fprintf(writeGrid, "2  %lf %lf %lf 0.0 0.0 0.0 \n", ((i*dx) + 3*dx), (yl + 3*dy), vx);
	//}
	for (int i = 0; i < (2 * mp) + 1; ++i) { //down
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", ((i * dx) + 2 * dx), (2 * dy));
	}
	for (int i = 0; i < (mp)+4; ++i) { //left
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", (2 * dx), ((i * dy) + (3 * dy)));
	}
	for (int i = 0; i < (mp)+5; ++i) { //right
		fprintf(writeGrid, "%lf %lf 0 0 0 0 \n", (2 * xl + 3 * dx), ((i * dy) + (2 * dy)));
	}

	//PARTICULAS TIPO 1 #fluido
	for (int i = 0; i < (2 * mp); i++) {
		for (int j = 0; j < np / 2; j++) {
			fprintf(writeGrid, "%lf %lf 2 0 0 0 \n", (i * dx + (3 * dx)), (j * dy + (3 * dy)));
		}
	}

	for (int i = 0; i < (2 * mp); i++) {
		for (int j = np / 2; j < np; j++) {
			fprintf(writeGrid, "%lf %lf 1 0 0 0 \n", (i * dx + (3 * dx)), (j * dy + (3 * dy)));
		}
	}

	fclose(writeGrid);
}

void createParticlesPoiseuilleFlow2D(double DL, double mult, const char filename[]) {

	int mp = 140 * mult;
	int np = 70*mult;

	double xl = mp * DL;
	double yl = np * DL;

	double dx = xl / mp;
	double dy = yl / np;

	FILE* writeGrid;
	fopen_s(&writeGrid,filename, "w");
	//fprintf(writeGrid, "0.0\n");
	fprintf(writeGrid, "%d \n", /*fluido*/ (np - 5) * (mp - 2) +/*first wall*/ +2 * mp /*second wall*/ + 2 * mp /*thrid wall*/ + 2 * mp /*+ /*fourth wall*/  /*+ /*fifth wall*/);

	//TIPO 3
	for (int i = 0; i < mp; i++) {
		fprintf(writeGrid, "%lf %lf -1 0 0 0 \n", i * dx);
	}

	for (int i = 0; i < mp; i++) {
		fprintf(writeGrid, "%lf %lf -1 0 0 0 \n", i * dx, dy * np);
	}

	//TIPO 2 
	for (int j = 1; j <= 2; j++) {
		for (int i = 0; i < mp; i++) {
			fprintf(writeGrid, "%lf %lf 0 -0.1 0 0 \n", i * dx, dy * j);
		}

		for (int i = 0; i < mp; i++) {
			fprintf(writeGrid, "%lf %lf 0 -0.1 0 0 \n", i * dx, dy * (np - j));
		}
	}

	//TIPO 0 e 4 (fluido)

	/*for (int j = 4; j < (np / 2); j++){
	for (int i = 0; i < mp; i++){
	fprintf(writeGrid, "0 %lf %lf 0\n", i*dx, dy*j);
	}
	}
	for (int j = 4; j <= np / 2; j++){
	for (int i = 0; i < mp; i++){
	fprintf(writeGrid, "0 %lf %lf 0\n", i*dx, dy*(np - j));
	}
	}*/

	for (int j = 3; j < (np - 2); j++) {
		for (int i = 1; i < mp - 1; i++) {
			fprintf(writeGrid, "%lf %lf 1 0.5 0 0 \n", i * dx, (dy * j)/* - (DL/2)*/);
		}
	}

	fclose(writeGrid);
}

void createParticlesDamBreak3D(double DL, double mult, const char filename[]) {

	int mp = 16*mult;
	int np = 8 *mult;
	int op = 8 *mult;

	double xl = mp * DL;
	double yl = np * DL;
	double zl = op * DL;

	double dx = xl / mp;
	double dy = yl / np;
	double dz = zl / op;

	//double vx = 0.0;

	FILE* writeGrid;
	fopen_s(&writeGrid, filename, "w");
	//fprintf(writeGrid, "0.0\n");
	fprintf(writeGrid, "%d \n", /*fluido*/(op) * ((mp)*np) +/*first wall*/op * ((4 * mp) + 10) + ((4 * mp) + 4) * (mp + 5) +/*second wall*/(op + 4) * ((4 * mp) + 14) + ((4 * mp) + 4) * (mp + 5) +/*thrid wall*/(op + 6) * ((4 * mp) + 18) + ((4 * mp) + 8) * (mp + 6) /*+ /*fourth wall (op + 8)*(4 * mp + 22) + (4 * mp + 12)*(mp + 7) + /*fifth wall (op + 10)*(4 * mp + 26) + (4 * mp + 16)*(mp + 8)*/);

	//PARTICULAS TIPO 3 #interna
	for (int k = 0; k < (op + 6); k++) {
		for (int i = 0; i < (2 * mp) + 6; ++i) { //down
			fprintf(writeGrid, "%lf 0.0 %lf -1 0.0 0.0 0.0 0.0 \n", (i * dx), dz * k - 3 * dz);
		}
		for (int i = 0; i < (mp)+6; ++i) { //left
			fprintf(writeGrid, "0.0 %lf %lf -1 0.0 0.0 0.0 0.0 \n", ((i * dy) + dy), dz * k - 3 * dz);
		}
		for (int i = 0; i < (mp)+6; ++i) { //right
			fprintf(writeGrid, "%lf %lf %lf -1 0.0 0.0 0.0 0.0 \n", (2 * xl + (5 * dx)), (i * dy + dy), dz * k - 3 * dz);
		}
	}

	for (int i = 0; i < ((2 * mp) + 4); ++i) { //front
		for (int j = 0; j < ((mp)+6); ++j) {
			fprintf(writeGrid, "%lf %lf %lf -1 0.0 0.0 0.0 0.0 \n", i * dx + dx, j * dy + dy, -3 * dz);
		}
	}
	for (int i = 0; i < ((2 * mp) + 4); ++i) { //back
		for (int j = 0; j < ((mp)+6); ++j) {
			fprintf(writeGrid, "%lf %lf %lf -1 0.0 0.0 0.0 0.0 \n", i * dx + dx, j * dy + dy, (op + 2) * dz);
		}
	}


	//PARTICULAS TIPO 2 #externa
	for (int k = 0; k < (op + 4); k++) {
		for (int i = 0; i < (2 * mp) + 4; ++i) { //down
			fprintf(writeGrid, "%lf %lf %lf 0 0.0 0.0 0.0 0.0 \n", ((i * dx) + dx), dy, dz * k - 2 * dz);
		}
		for (int i = 0; i < (mp)+5; ++i) { //left
			fprintf(writeGrid, "%lf %lf %lf 0 0.0 0.0 0.0 0.0 \n", dx, ((i * dy) + (2 * dy)), dz * k - 2 * dz);
		}
		for (int i = 0; i < (mp)+5; ++i) { //right
			fprintf(writeGrid, "%lf %lf %lf 0 0.0 0.0 0.0 0.0 \n", (2 * xl + (4 * dx)), ((i * dy) + 2 * dy), dz * k - 2 * dz);
		}
	}

	for (int i = 0; i < ((2 * mp) + 2); ++i) { //front
		for (int j = 0; j < ((mp)+5); ++j) {
			fprintf(writeGrid, "%lf %lf %lf 0 0.0 0.0 0.0 0.0 \n", ((i * dx) + 2 * dx), (2 * dy + j * dy), -2 * dz);
		}
	}
	for (int i = 0; i < ((2 * mp) + 2); ++i) { //back
		for (int j = 0; j < ((mp)+5); ++j) {
			fprintf(writeGrid, "%lf %lf %lf 0 0.0 0.0 0.0 0.0 \n", ((i * dx) + 2 * dx), (2 * dy + j * dy), (op + 1) * dz);
		}
	}

	//PARTICULAS TIPO 2 #interna
	for (int k = 0; k < (op); k++) {
		for (int i = 0; i < (2 * mp) + 1; ++i) { //down
			fprintf(writeGrid, "%lf %lf %lf 0 0.0 0.0 0.0 0.0 \n", ((i * dx) + 2 * dx), (2 * dy), dz * k);
		}
		for (int i = 0; i < (mp)+4; ++i) { //left
			fprintf(writeGrid, "%lf %lf %lf 0 0.0 0.0 0.0 0.0 \n", (2 * dx), ((i * dy) + (3 * dy)), dz * k);
		}
		for (int i = 0; i < (mp)+5; ++i) { //right
			fprintf(writeGrid, "%lf %lf %lf 0 0.0 0.0 0.0 0.0 \n", (2 * xl + 3 * dx), ((i * dy) + (2 * dy)), dz * k);
		}
	}
	for (int i = 0; i < ((2 * mp) + 2); ++i) { //front
		for (int j = 0; j < ((mp)+5); ++j) {
			fprintf(writeGrid, "%lf %lf %lf 0 0.0 0.0 0.0 0.0 \n", ((i * dx) + 2 * dx), ((2 * dy) + j * dy), -dz);
		}
	}
	for (int i = 0; i < ((2 * mp) + 2); ++i) { //back
		for (int j = 0; j < ((mp)+5); ++j) {
			fprintf(writeGrid, "%lf %lf %lf 0 0.0 0.0 0.0 0.0 \n", ((i * dx) + 2 * dx), ((2 * dy) + j * dy), (op)*dz);
		}
	}

	//PARTICULAS TIPO 0 #fluido
	for (int k = 0; k < (op); k++) {
		for (int i = 0; i < (mp / 2); i++) {
			for (int j = 0; j < (2 * np); j++) {
				fprintf(writeGrid, "%lf %lf %lf 1 0.0 0.0 0.0 0.0 \n", (i * dx + (3 * dx)), (j * dy + (3 * dy)), dz * k);
			}
		}
	}

	fclose(writeGrid);
}

void createParticlesDamBreak3D_offset_nozero(double DL, double mult, const char filename[]) {

	int mp = 10*mult;
	int np = 5* mult;
	int op = 5* mult;

	double xl = mp * DL;
	double yl = np * DL;
	double zl = op * DL;

	double dx = xl / mp;
	double dy = yl / np;
	double dz = zl / op;

	double offsetX = 0 * dx;
	double offsetY = 0 * dx;
	double offsetZ = 0 * dx;

	FILE* writeGrid;
	fopen_s(&writeGrid, filename, "w");
	//fprintf(writeGrid, "0.0\n");
	fprintf(writeGrid, "%d \n", /*fluido*/(op) * (mp * np / 2) +/*first wall*/op * (4 * mp + 10) + (4 * mp /*+ 4*/) * (mp /*+ 5*/ + 4) +/*second wall*/(op + 4) * (4 * mp + 14) + (4 * mp + 4) * (mp + 5) +/*thrid wall*/(op + 6) * (4 * mp + 18) + (4 * mp + 8) * (mp + 6) /*+ /*fourth wall (op + 8)*(4 * mp + 22) + (4 * mp + 12)*(mp + 7) + /*fifth wall (op + 10)*(4 * mp + 26) + (4 * mp + 16)*(mp + 8)*/);

	//PARTICULAS TIPO -1 #interna
	for (int k = 0; k < (op + 6); k++) {
		for (int i = 0; i < (2 * mp) + 6; ++i) { //down
			fprintf(writeGrid, "%lf %lf %lf -1 0 0 0 0 \n", (i * dx) + offsetX, offsetY, dz * k - 3 * dz + offsetZ);
		}
		for (int i = 0; i < (mp)+6; ++i) { //left
			fprintf(writeGrid, "%lf %lf %lf -1 0 0 0 0 \n", offsetX, ((i * dy) + dy) + offsetY, dz * k - 3 * dz + offsetZ);
		}
		for (int i = 0; i < (mp)+6; ++i) { //right
			fprintf(writeGrid, "%lf %lf %lf -1 0 0 0 0 \n", (2 * xl + (5 * dx)) + offsetX, (i * dy + dy) + offsetY, dz * k - 3 * dz + offsetZ);
		}
	}

	for (int i = 0; i < ((2 * mp) + 4); ++i) { //front
		for (int j = 0; j < ((mp)+6); ++j) {
			fprintf(writeGrid, "%lf %lf %lf -1 0 0 0 0 \n", i * dx + dx + offsetX, j * dy + dy + offsetY, -3 * dz + offsetZ);
		}
	}
	for (int i = 0; i < ((2 * mp) + 4); ++i) { //back
		for (int j = 0; j < ((mp)+6); ++j) {
			fprintf(writeGrid, "%lf %lf %lf -1 0 0 0 0 \n", i * dx + dx + offsetX, j * dy + dy + offsetY, (op + 2) * dz + offsetZ);
		}
	}


	//PARTICULAS TIPO 0 #externa
	for (int k = 0; k < (op + 4); k++) {
		for (int i = 0; i < (2 * mp) + 4; ++i) { //down
			fprintf(writeGrid, "%lf %lf %lf 0 0 0 0 0 \n", ((i * dx) + dx) + offsetX, dy + offsetY, dz * k - 2 * dz + offsetZ);
		}
		for (int i = 0; i < (mp)+5; ++i) { //left
			fprintf(writeGrid, "%lf %lf %lf 0 0 0 0 0 \n", dx + offsetX, ((i * dy) + (2 * dy)) + offsetY, dz * k - 2 * dz + offsetZ);
		}
		for (int i = 0; i < (mp)+5; ++i) { //right
			fprintf(writeGrid, "%lf %lf %lf 0 0 0 0 0 \n", (2 * xl + (4 * dx)) + offsetX, ((i * dy) + 2 * dy) + offsetY, dz * k - 2 * dz + offsetZ);
		}
	}

	for (int i = 0; i < ((2 * mp) + 2); ++i) { //front
		for (int j = 0; j < ((mp)+5); ++j) {
			fprintf(writeGrid, "%lf %lf %lf 0 0 0 0 0 \n", ((i * dx) + 2 * dx) + offsetX, (2 * dy + j * dy) + offsetY, -2 * dz + offsetZ);
		}
	}
	for (int i = 0; i < ((2 * mp) + 2); ++i) { //back
		for (int j = 0; j < ((mp)+5); ++j) {
			fprintf(writeGrid, "%lf %lf %lf 0 0 0 0 0 \n", ((i * dx) + 2 * dx) + offsetX, (2 * dy + j * dy) + offsetY, (op + 1) * dz + offsetZ);
		}
	}

	//PARTICULAS TIPO 0 #interna
	for (int k = 0; k < (op); k++) {
		for (int i = 0; i < (2 * mp) + 1; ++i) { //down
			fprintf(writeGrid, "%lf %lf %lf 0 0 0 0 0 \n", ((i * dx) + 2 * dx) + offsetX, (2 * dy) + offsetY, dz * k + offsetZ);
		}
		for (int i = 0; i < (mp)+4; ++i) { //left
			fprintf(writeGrid, "%lf %lf %lf 0 0 0 0 0 \n", (2 * dx) + offsetX, ((i * dy) + (3 * dy)) + offsetY, dz * k + offsetZ);
		}
		for (int i = 0; i < (mp)+5; ++i) { //right
			fprintf(writeGrid, "%lf %lf %lf 0 0 0 0 0 \n", (2 * xl + 3 * dx) + offsetX, ((i * dy) + (2 * dy)) + offsetY, dz * k + offsetZ);
		}
	}
	for (int i = 0; i < ((2 * mp) /*+ 2*/); ++i) { //front
		for (int j = 0; j < ((mp)/*+5*/ +4); ++j) {
			fprintf(writeGrid, "%lf %lf %lf 0 0 0 0 0 \n", ((i * dx) + 2 * dx) + offsetX, ((2 * dy) + j * dy) + offsetY, -dz + offsetZ);
		}
	}
	for (int i = 0; i < ((2 * mp) /*+ 2*/); ++i) { //back
		for (int j = 0; j < ((mp)/*+5*/+4); ++j) {
			fprintf(writeGrid, "%lf %lf %lf 0 0 0 0 0 \n", ((i * dx) + 2 * dx) + offsetX, ((2 * dy) + j * dy) + offsetY, (op)*dz + offsetZ);
		}
	}

	//PARTICULAS TIPO 1 #fluido
	for (int k = 0; k < (op); k++) {
		for (int i = 0; i < (mp / 2); i++) {
			for (int j = 0; j < (np); j++) {
				fprintf(writeGrid, "%lf %lf %lf 1 0 0 0 0 \n", (i * dx + (3 * dx)) + offsetX, (j * dy + (3 * dy)) + offsetY, dz * k + offsetZ);
			}
		}
	}

	////PARTICULAS TIPO 2 #fluido
	//for (int k = 0; k < (op); k++) {
	//	for (int i = 0; i < (mp / 2); i++) {
	//		for (int j = 0; j < (np); j++) {
	//			fprintf(writeGrid, "%lf %lf %lf 2 0 0 0 0 \n", (i*dx + (3 * dx)) + offsetX, (j*dy + (3 * dy)) + offsetY, dz*k + offsetZ);
	//		}
	//	}
	//}

	fclose(writeGrid);
}