/*

Voxar-MPS (Fluid simulation framework based on the MPS method)
Copyright (C) 2007-2011  Ahmad Shakibaeinia
Copyright (C) 2016-2019  Voxar Labs

Voxar-MPS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


*/

#include "inOut.h"

void getTP(int& TP)
{
	ifstream in;
	string filename = "inputs/in.vtu";
	in.open(filename, ios::out);
	while (!in.is_open())
	{
		cout << "Trying to read from vtu file" << endl;
	}
	in.ignore(256, '\n');
	in.ignore(256, '\n');
	in.ignore(256, '\n');
	in.ignore(256, '\'');
	in >> TP;
	in.close();
}

void getVTU(double*& x, double*& y, double*& z, double*& unew, double*& vnew, double*& wnew, double*& pnew, int*& PTYPE, int& TP, int& FP, int& GP, int& WP,
	double& Xmin, double& Xmax, double& Ymin, double& Ymax, double& Zmin, double& Zmax)
{

	int b;
	string test;
	ifstream in;
	string filename = "inputs/in.vtu";
	in.open(filename, ios::out);
	while (!in.is_open())
	{
		cout << "Trying to read from vtu file" << endl;
	}
	in.ignore(256, '\n');
	in.ignore(256, '\n');
	in.ignore(256, '\n');
	in.ignore(256, '\'');
	in >> TP;
	int nump = TP;
	in.ignore(256, '\n');
	in.ignore(256, '\n');
	in.ignore(256, '\n');
	for (b = 1; b <= nump; b++)
	{
		in >> x[b] >> y[b] >> z[b];
		if (x[b] < Xmin) Xmin = x[b];
		if (x[b] > Xmax) Xmax = x[b];
		if (y[b] < Ymin) Ymin = y[b];
		if (y[b] > Ymax) Ymax = y[b];
		if (z[b] < Zmin) Zmin = z[b];
		if (z[b] > Zmax) Zmax = z[b];
	}

	in.ignore(256, '\n');
	in.ignore(256, '\n');
	in.ignore(256, '\n');
	in.ignore(256, '\n');
	in.ignore(256, '\n');
	for (b = 1; b <= nump; b++)
	{
		in >> unew[b] >> vnew[b] >> wnew[b];
	}
	in.ignore(256, '\n');
	in.ignore(256, '\n');
	in.ignore(256, '\n');
	for (b = 1; b <= nump; b++)
	{
		in >> pnew[b];
	}
	in.ignore(256, '\n');
	in.ignore(256, '\n');
	in.ignore(256, '\n');
	for (b = 1; b <= nump; b++)
	{
		in >> PTYPE[b];
		if (PTYPE[b] < 0)  GP++;
		if (PTYPE[b] == 0) WP++;
		if (PTYPE[b] > 0)  FP++;
	}
	in.close();
}

void saveParticles(int TP, int Tstep, double* x, double* y, double* z, double* unew, double* vnew, double* wnew, double* pnew, int* PTYPE, int dim)
{
	int nump = TP, number = Tstep;
	ofstream out;

	if (!(CreateDirectoryA("output", NULL) || ERROR_ALREADY_EXISTS == GetLastError()))
	{
		cout << "Inexistent directory\n";
		system("pause");
	}

	string filename = "output/out" + to_string(number) + ".vtu";
	out.open(filename, ios::out);
	while (!out.is_open())
	{
		cout << "Trying to write to vtu file" << endl;
	}

	out << "<?xml version='1.0' encoding='UTF-8'?>\n";
	out << "<VTKFile xmlns='VTK' byte_order='LittleEndian' version='0.1' type='UnstructuredGrid'>\n";
	out << " <UnstructuredGrid>\n";

	out << "  <Piece NumberOfCells='" << nump << "' NumberOfPoints='" << nump << "'>\n";

	out << "   <Points>\n";
	out << "    <DataArray NumberOfComponents='3' type='Float32' Name='Position' format='ascii'>\n";

	int b;
	if (dim == 3)
	{
		for (b = 1; b <= nump; b++)
			out << x[b] << " " << y[b] << " " << z[b] << " ";
	}
	else
	{
		for (b = 1; b <= nump; b++)
		{
			out << x[b] << " " << y[b] << " " << 0 << " ";
		}
	}
	out << "\n";
	out << "    </DataArray>\n";
	out << "   </Points>\n";

	out << "   <PointData>\n";

	out << "    <DataArray NumberOfComponents='3' type='Float32' Name='Velocity' format='ascii'>\n";
	if (dim == 3)
	{
		for (b = 1; b <= nump; b++)
			out << unew[b] << " " << vnew[b] << " " << wnew[b] << " ";
	}
	else
	{
		for (b = 1; b <= nump; b++)
			out << unew[b] << " " << vnew[b] << " " << 0 << " ";
	}

	out << "\n";
	out << "    </DataArray>\n";

	out << "    <DataArray NumberOfComponents='1' type='Float32' Name='Pressure' format='ascii'>\n";


	for (b = 1; b <= nump; b++)
		out << pnew[b] << " ";

	out << "\n";
	out << "    </DataArray>\n";

	out << "    <DataArray NumberOfComponents='1' type='Int32' Name='partType' format='ascii'>\n";


	for (b = 1; b <= nump; b++)
		out << PTYPE[b] << " ";

	out << "\n";
	out << "    </DataArray>\n";

	out << "   </PointData>\n";

	//VTK specific information
	out << "   <Cells>\n";

	//Connectivity
	out << "    <DataArray type='Int32' Name='connectivity' format='ascii'>\n";
	for (b = 0; b < nump; b++)
	{
		out << b << " ";
	}
	out << "\n";
	out << "    </DataArray>\n";

	//Offsets
	out << "    <DataArray type='Int32' Name='offsets' format='ascii'>\n";
	for (b = 1; b <= nump; b++)
		out << b << " ";

	out << "\n";
	out << "    </DataArray>\n";

	//Types
	out << "    <DataArray type='UInt8' Name='types' format='ascii'>\n";
	for (b = 0; b < nump; b++)
		out << "1 ";

	out << "\n";
	out << "    </DataArray>\n";

	out << "   </Cells>\n";

	out << "  </Piece>\n";
	out << " </UnstructuredGrid>\n";
	out << "</VTKFile>";

	out.close();
}

void saveFluidParticles(int TP, int FP, int Tstep, double* x, double* y, double* z, double* unew, double* vnew, double* wnew, double* pnew, int* PTYPE, int dim)
{
	int nump = TP, number = Tstep;
	ofstream out;

	if (!(CreateDirectoryA("output_fluid", NULL) || ERROR_ALREADY_EXISTS == GetLastError()))
	{
		cout << "Inexistent directory\n";
		system("pause");
	}

	string filename = "output_fluid/out" + to_string(number) + ".vtu";
	out.open(filename, ios::out);
	while (!out.is_open())
	{
		cout << "Trying to write to vtu file" << endl;
	}

	out << "<?xml version='1.0' encoding='UTF-8'?>\n";
	out << "<VTKFile xmlns='VTK' byte_order='LittleEndian' version='0.1' type='UnstructuredGrid'>\n";
	out << " <UnstructuredGrid>\n";

	out << "  <Piece NumberOfCells='" << FP << "' NumberOfPoints='" << FP << "'>\n";

	out << "   <Points>\n";
	out << "    <DataArray NumberOfComponents='3' type='Float32' Name='Position' format='ascii'>\n";

	int b;
	if (dim == 3)
	{
		for (b = (TP - FP) + 1; b <= nump; b++)
			out << x[b] << " " << y[b] << " " << z[b] << " ";
	}
	else
	{
		for (b = (TP - FP) + 1; b <= nump; b++)
			out << x[b] << " " << y[b] << " " << 0 << " ";
	}
	out << "\n";
	out << "    </DataArray>\n";
	out << "   </Points>\n";

	out << "   <PointData>\n";

	out << "    <DataArray NumberOfComponents='3' type='Float32' Name='Velocity' format='ascii'>\n";
	if (dim == 3)
	{
		for (b = (TP - FP) + 1; b <= nump; b++)
			out << unew[b] << " " << vnew[b] << " " << wnew[b] << " ";
	}
	else
	{
		for (b = (TP - FP) + 1; b <= nump; b++)
			out << unew[b] << " " << vnew[b] << " " << 0 << " ";
	}

	out << "\n";
	out << "    </DataArray>\n";

	out << "    <DataArray NumberOfComponents='1' type='Float32' Name='Pressure' format='ascii'>\n";


	for (b = (TP - FP) + 1; b <= nump; b++)
		out << pnew[b] << " ";

	out << "\n";
	out << "    </DataArray>\n";

	out << "    <DataArray NumberOfComponents='1' type='Int32' Name='partType' format='ascii'>\n";


	for (b = (TP - FP) + 1; b <= nump; b++)
		out << PTYPE[b] << " ";

	out << "\n";
	out << "    </DataArray>\n";

	out << "   </PointData>\n";

	//VTK specific information
	out << "   <Cells>\n";

	//Connectivity
	out << "    <DataArray type='Int32' Name='connectivity' format='ascii'>\n";
	for (b = 0; b < FP; b++)
	{
		out << b << " ";
	}
	out << "\n";
	out << "    </DataArray>\n";

	//Offsets
	out << "    <DataArray type='Int32' Name='offsets' format='ascii'>\n";
	for (b = 1; b <= FP; b++)
		out << b << " ";

	out << "\n";
	out << "    </DataArray>\n";

	//Types
	out << "    <DataArray type='UInt8' Name='types' format='ascii'>\n";
	for (b = 0; b < FP; b++)
		out << "1 ";

	out << "\n";
	out << "    </DataArray>\n";

	out << "   </Cells>\n";

	out << "  </Piece>\n";
	out << " </UnstructuredGrid>\n";
	out << "</VTKFile>";

	out.close();
}

