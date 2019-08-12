using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace GUI
{
    public partial class Form1 : Form
    {
        string voxarMpsGui;

        public Form1()
        {
            InitializeComponent();
            comboBox1.Items.Add("Português-BR");
            comboBox1.Items.Add("English-US");
            comboBox1.Text = comboBox1.Items[0].ToString();

            voxarMpsGui = "";
            this.FormClosed += new FormClosedEventHandler(Form1_FormClosed);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if(openFileDialog1.ShowDialog(this) == DialogResult.OK)
            {
                System.IO.FileInfo fInfo = new System.IO.FileInfo(openFileDialog1.FileName);

                string strFileName = fInfo.Name;

                string strFilePath = fInfo.DirectoryName;

                //textBox1.Text = strFilePath + "\\" + strFileName;
            }
        }


        private void button3_Click(object sender, EventArgs e)
        {

            if (comboBox2.SelectedItem == null || comboBox3.SelectedItem == null || comboBox4.SelectedItem == null || comboBox5.SelectedItem == null 
                || numericUpDown1.Value.ToString() == string.Empty || numericUpDown2.Value.ToString() == string.Empty || numericUpDown3.Value.ToString() == string.Empty
                || numericUpDown4.Value.ToString() == string.Empty || numericUpDown5.Value.ToString() == string.Empty || numericUpDown6.Value.ToString() == string.Empty)
            {
                if (comboBox1.SelectedItem.ToString() == "Português-BR")
                {
                    MessageBox.Show(this, "Verifique os campos preenchidos", "Erro");
                }
                else if (comboBox1.SelectedItem.ToString() == "English-US")
                {
                    MessageBox.Show(this, "Check the fields", "Error");
                }
                return;
            }

            if (decimal.ToDouble(numericUpDown6.Value) > 0.005)
            {
                if (comboBox1.SelectedItem.ToString() == "Português-BR")
                {
                    MessageBox.Show(this, "A maior duração permitida do passo de tempo é 0.005 segundos, caso contrário a simulação apresentará instabilidades significantes", "Erro");
                }
                else if (comboBox1.SelectedItem.ToString() == "English-US")
                {
                    MessageBox.Show(this, "The largest time-step duration value allowed is 0.005 seconds, ", "Error");
                }
                return;
            }
            

            string codeOpt = "";
            if (comboBox3.SelectedItem.ToString() == "GPU-CUDA (faster)" || comboBox3.SelectedItem.ToString() == "GPU-CUDA (mais rápido)")
            {
                codeOpt = "gpu";
            }
            else if (comboBox3.SelectedItem.ToString() == "OpenMP")
            {
                codeOpt = "openmp";
            }
            else if (comboBox3.SelectedItem.ToString() == "Sequential" || comboBox3.SelectedItem.ToString() == "Sequencial")
            {
                codeOpt = "sequential";
            }
            else { codeOpt = "notright"; MessageBox.Show(this, "Contact developer, codeOpt not right.", "Error"); }

            string testCase = "";

            if (comboBox4.SelectedItem.ToString() == "Dam Break" || comboBox4.SelectedItem.ToString() == "Quebra de Barragem")
            {
                testCase = "dam";
            }
            else if (comboBox4.SelectedItem.ToString() == "Water Drop (only 2D)" || comboBox4.SelectedItem.ToString() == "Gota d'água (apenas 2D)")
            {
                testCase = "drop";
            }
            else if (comboBox4.SelectedItem.ToString() == "Poiseuille Flow (only 2D)" || comboBox4.SelectedItem.ToString() == "Escoamento de Poiseuille (apenas 2D)")
            {
                testCase = "turb";
            }
            else if (comboBox4.SelectedItem.ToString() == "Oil Spilling (only 2D)" || comboBox4.SelectedItem.ToString() == "Derramamento de óleo (apenas 2D)")
            {
                testCase = "oil";
            }
            else if (comboBox4.SelectedItem.ToString() == "Shear Cavity (only 2D)" || comboBox4.SelectedItem.ToString() == "Escoamento da Cavidade (apenas 2D)")
            {
                testCase = "shear";
            }
            else { testCase = "notright"; MessageBox.Show(this, "Contact developer, testCase not right.", "Error"); }

            string comp = "";
            if (comboBox5.SelectedItem.ToString() == "Weakly Compressible" || comboBox5.SelectedItem.ToString() == "Fracamente Compressível")
            {
                comp = "comp";
            }
            else if (comboBox5.SelectedItem.ToString() == "Incompressible (more precise)" || comboBox5.SelectedItem.ToString() == "Incompressível (mais preciso)")
            {
                comp = "incomp";
            }
            else { comp = "notright"; MessageBox.Show(this, "Contact developer, comp not right.", "Error"); }

            string turb = "";
            if (checkBox1.Checked) turb = "turb";
            else turb = "noturb";

            string visc = "";
            if (checkBox2.Checked) visc = "visc";
            else visc = "novisc";

            string matopt = "";
            if (checkBox3.Checked) matopt = "matopt";
            else matopt = "nomatopt";

            string srcopt = "";
            if (checkBox4.Checked) srcopt = "srcopt";
            else srcopt = "nosrcopt";

            string printall = "";
            if (checkBox5.Checked) printall = "printall";
            else printall = "no";

            string printfluid = "";
            if (checkBox6.Checked) printfluid = "printfluid";
            else printfluid = "no";

            if(printall != "printall" && printfluid != "printfluid")
            {
                if (comboBox1.SelectedItem.ToString() == "Português-BR")
                {
                    MessageBox.Show(this, "Se nenhuma opção de saída for selecionada, a simulação não poderá ser visualizada", "Aviso");
                }
                else if (comboBox1.SelectedItem.ToString() == "English-US")
                {
                    MessageBox.Show(this, "If no output option is selected, the simulation will not be able to be visualized", "Warning");
                }
            }

            var proc = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "..\\..\\..\\x64\\Release\\VoxarMPS.exe",
                    Arguments = testCase + " " + visc + " " + turb + " " + comp + " " + matopt + " " + srcopt + " " + codeOpt + " " + comboBox2.SelectedItem.ToString() + " " + printall + " " + printfluid +
                    " " + numericUpDown1.Value + " " + numericUpDown2.Value + " " + numericUpDown3.Value + " " + numericUpDown4.Value + " " + numericUpDown5.Value + " " + numericUpDown6.Value,
                    //Arguments = "dam visc turb comp matopt srcopt gpu 2d printall printfluid ...",
                    UseShellExecute = true,
                    RedirectStandardOutput = false,
                    CreateNoWindow = false
                }
            };
            Console.WriteLine(proc.StartInfo.Arguments);
            proc.Start();
        }

        void Form1_FormClosed(object sender, FormClosedEventArgs e)
        {
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (comboBox1.SelectedItem.ToString() == "Português-BR")
            {
                putInPortuguese();
            }
            else if (comboBox1.SelectedItem.ToString() == "English-US")
            {
                putInEnglish();
            }
        }

        public void putInEnglish()
        {
            this.button3.Text = "Start";
            this.label13.Text = "Idioma:";
            this.label14.Text = "Performance optimization:";
            this.label15.Text = "Dimensions:";
            this.label1.Text = "Test:";
            this.label6.Text = "Compressibility:";
            this.comboBox3.Items[2] = "Sequential";
            this.comboBox3.Items[0] = "GPU-CUDA (faster)";
            this.comboBox4.Items[0] = "Dam Break";
            this.comboBox4.Items[1] = "Oil Spilling (only 2D)";
            this.comboBox4.Items[2] = "Water Drop (only 2D)";
            this.comboBox4.Items[3] = "Poiseuille Flow (only 2D)";
            this.comboBox4.Items[4] = "Shear Cavity (only 2D)";
            this.comboBox5.Items[0] = "Weakly Compressible";
            this.comboBox5.Items[1] = "Incompressible (more precise)";
            this.checkBox1.Text = "Turbulence";
            this.checkBox2.Text = "Viscoplasticity";
            this.checkBox3.Text = "More stable fluid pressure";
            this.checkBox4.Text = "Better conservation of the fluid's angular \n and linear momentum (recommended)";
            this.checkBox5.Text = "Show all particles";
            this.checkBox6.Text = "Show only fluid particles";
            this.groupBox1.Text = "Incompressibility Options";
            this.groupBox2.Text = "Output Options";
            this.label2.Text = "Density (Fluid 1):";
            this.label3.Text = "Density (Fluid 2):";
            this.label4.Text = "Viscosity (Fluid 1):";
            this.label5.Text = "Viscosity (Fluid 2):";
            this.label17.Text = "Time-step duration:";
            this.label7.Text = "Total simulation time:";
            this.label8.Text = "seconds";
            this.label16.Text = "seconds";
        }

        public void putInPortuguese()
        {
            this.button3.Text = "Iniciar";
            this.label13.Text = "Language:";
            this.label14.Text = "Otimização de performance:";
            this.label15.Text = "Dimensões:";
            this.label1.Text = "Teste:";
            this.label6.Text = "Compressibilidade:";
            this.comboBox3.Items[2] = "Sequencial";
            this.comboBox3.Items[0] = "GPU-CUDA (mais rápido)";
            this.comboBox4.Items[0] = "Quebra de Barragem";
            this.comboBox4.Items[1] = "Derramamento de óleo (apenas 2D)";
            this.comboBox4.Items[2] = "Gota d'água (apenas 2D)";
            this.comboBox4.Items[3] = "Escoamento de Poiseuille (apenas 2D)";
            this.comboBox4.Items[4] = "Escoamento da Cavidade (apenas 2D)";
            this.comboBox5.Items[0] = "Fracamente Compressível";
            this.comboBox5.Items[1] = "Incompressível (mais preciso)";
            this.checkBox1.Text = "Turbulência";
            this.checkBox2.Text = "Viscoplasticidade";
            this.checkBox3.Text = "Maior estabilidade da pressão do fluido";
            this.checkBox4.Text = "Maior conservação do momento angular \n e linear do fluido (recomendado)";
            this.checkBox5.Text = "Mostrar todas as partículas";
            this.checkBox6.Text = "Mostrar apenas partículas de fluido";
            this.groupBox1.Text = "Opções de Incompressibilidade";
            this.groupBox2.Text = "Opções de Saída";
            this.label2.Text = "Densidade (Fluido 1):";
            this.label3.Text = "Densidade (Fluido 2):";
            this.label4.Text = "Viscosidade (Fluido 1):";
            this.label5.Text = "Viscosidade (Fluido 2):";
            this.label7.Text = "Tempo total de simulação:";
            this.label17.Text = "Duração do passo de tempo:";
            this.label8.Text = "segundos";
            this.label16.Text = "segundos";

        }

        private void button2_Click(object sender, EventArgs e)
        {
            bool isRunning = Process.GetProcessesByName(voxarMpsGui).FirstOrDefault(p => p.ProcessName.Equals(voxarMpsGui)) != default(Process);

            if (!isRunning)
            {

                var proc = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "./Python35/pyportable.exe",
                        WorkingDirectory = @"./voxarMPS/",
                        Arguments = "-m http.server",
                        UseShellExecute = false,
                        RedirectStandardOutput = false,
                        CreateNoWindow = false
                    }
                };
                proc.Start();
                voxarMpsGui = proc.ProcessName;

            }

            System.Diagnostics.Process.Start("http://localhost:8000");
        }

        private void checkBox2_CheckedChanged(object sender, EventArgs e)
        {
            //if (checkBox2.Checked) panel1.Enabled = true;
            //else panel1.Enabled = false;
        }

        private void comboBox5_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (comboBox5.SelectedIndex != 0) groupBox1.Enabled = true;
            else groupBox1.Enabled = false;
        }

        private void comboBox4_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (comboBox4.SelectedIndex == 1) {
                numericUpDown3.Enabled = true;
                numericUpDown4.Enabled = true;
                checkBox2.Enabled = true;
            } else {
                numericUpDown3.Enabled = false;
                numericUpDown4.Enabled = false;
                checkBox2.Enabled = false;
            }
        }

    }
}
