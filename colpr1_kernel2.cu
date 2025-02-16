extern "C" {
__device__ double log10(double x) {
    return log(x) / log(10.0);
}

__device__ void gauss3(double* a, double* b, int nd, int ndim) {
    a[0 * 3 + 2] /= a[0 * 3 + 1];
    b[0] /= a[0 * 3 + 1];
    a[0 * 3 + 1] = 1.0;

    for (int j = 1; j < nd; ++j) {
        double piv = a[j * 3 + 0];
        if (piv != 0) {
            for (int k = 0; k < 3; ++k) {
                a[j * 3 + k] /= piv;
            }
            b[j] /= piv;
            a[j * 3 + 1] -= a[(j - 1) * 3 + 2];
            a[j * 3 + 0] -= a[(j - 1) * 3 + 1];
            b[j] -= b[j - 1];
        }
        b[j] /= a[j * 3 + 1];
        a[j * 3 + 2] /= a[j * 3 + 1];
        a[j * 3 + 1] = 1.0;
    }

    for (int j = nd - 2; j >= 0; --j) {
        if (a[j * 3 + 2] != 0) {
            b[j] -= b[j + 1] * a[j * 3 + 2];
        }
        b[j] /= a[j * 3 + 1];
        a[j * 3 + 1] = 1.0;
    }
}

__device__ void spline(double* x, double* g, double* s, double* a, double* b, double* c,
                        double* d, int maxdat) {
    for (int i = 1; i < maxdat - 1; ++i) {
        double alfap = (x[i + 1] - x[i]) / 3.0;
        double alfam = (x[i] - x[i - 1]) / 3.0;
        double betap = (g[i + 1] - g[i]) / (x[i + 1] - x[i]);
        double betam = (g[i] - g[i - 1]) / (x[i] - x[i - 1]);
        a[(i - 1) * 3 + 0] = alfam;
        a[(i - 1) * 3 + 2] = alfap;
        a[(i - 1) * 3 + 1] = (alfam + alfap) * 2.0;
        s[i - 1] = betap - betam;
    }

    a[0 * 3 + 0] = 0.0;
    a[(maxdat - 3) * 3 + 2] = 0.0;
    c[0] = ((g[2] - g[0]) / (x[2] - x[0]) - (g[1] - g[0]) / (x[1] - x[0])) / (x[2] - x[1]);
    int i = maxdat - 1;
    c[i] = ((g[i] - g[i - 2]) / (x[i] - x[i - 2]) - (g[i - 1] - g[i - 2]) / (x[i - 1] - x[i - 2])) / (x[i] - x[i - 1]);
    s[0] -= c[0] * (x[1] - x[0]) / 3.0;
    s[maxdat - 3] -= c[maxdat - 1] * (x[maxdat - 1] - x[maxdat - 2]) / 3.0;
    
    // Perform Gaussian elimination
    gauss3(a, s, maxdat - 2, 10);

    for (int i = 0; i < maxdat - 2; ++i) {
        c[i + 1] = s[i];
    }

    for (int i = 0; i < maxdat - 1; ++i) {
        d[i] = (c[i + 1] - c[i]) / (x[i + 1] - x[i]) / 3.0;
        b[i] = (g[i + 1] - g[i]) / (x[i + 1] - x[i]) - (c[i + 1] + 2.0 * c[i]) * (x[i + 1] - x[i]) / 3.0;
    }

    b[maxdat - 1] = b[maxdat - 2] + (c[maxdat - 2] + c[maxdat - 1]) * (x[maxdat - 1] - x[maxdat - 2]);
    d[maxdat] = 0.0;
}

__device__ double inter1(int ig, int nreac, int nnuc, double vsig0, double temp1, int mxsig0,
                        double* data_x, double* data_tab, int* data_idnuc1, int* data_msf,
                        double* data_g, double* data_ftab, double* data_gg, int maxdat,
                        double* data_s, double* data_a, double* data_b, double* data_c,
                        double* data_d, double* data_bg, double* data_cg, double* data_dg,
                        double* data_vt, double* data_tt, double* data_ft) {
    for (int i = 0; i < mxsig0; ++i) {
        data_x[i] = log10(data_tab[i * 391 + nnuc]);
    }

    int match = 0;
    double vlsig0 = log10(vsig0);
    int nncomp = data_idnuc1[nnuc];

    for (int i = 0; i < mxsig0 - 1; ++i) {
        if (vlsig0 < data_x[i + 1]) {
            match = i;
            break;
        }
    }

    for (int j = 0; j < data_msf[nreac * 391 + nnuc]; ++j) {
        for (int i = 0; i < mxsig0; ++i) {
            data_g[i] = data_ftab[i * 4 * 70 * 6 * 41 + j * 70 * 6 * 41 + ig * 6 * 41 + nreac * 41 + nncomp];
            data_gg[i * 5 + j] = data_ftab[i * 4 * 70 * 6 * 41 + j * 70 * 6 * 41 + ig * 6 * 41 + nreac * 41 + nncomp];
        }
        maxdat = mxsig0;
        spline(data_x, data_g, data_s, data_a, data_b, data_c, data_d, maxdat);
        for (int i = 0; i < mxsig0; ++i) {
            data_bg[i * 5 + j] = data_b[i];
            data_cg[i * 5 + j] = data_c[i];
            data_dg[i * 5 + j] = data_d[i];
        }
        double dxx = vlsig0 - data_x[match];
        data_vt[j] = data_gg[match * 5 + j] + data_bg[match * 5 + j] * dxx + data_cg[match * 5 + j] \
                    * dxx * dxx + data_dg[match * 5 + j] * dxx * dxx * dxx;
    }

    for (int i = 0; i < data_msf[nreac * 391 + nnuc]; ++i) {
        data_tt[i] = data_ft[i * 391 + nnuc];
    }

    double fval = data_vt[0]; // skip splan3
    return fval;
}

__device__ void calsg2(double* data_rd, int nfubnd, int mxrg, int mxncfu, const int* mnccel,
                        const int* data_idnc1, const double* data_adnc, const double* data_volrg,
                        double volrgt, const double* data_sigt, double fbell, const double* data_atden,
                        double* data_sig0) {
    
    double s0pv0 = 2.0 / data_rd[nfubnd - 1];
    double s0pv2 = 2.0 * data_rd[nfubnd - 1] / (data_rd[mxrg - 1] * data_rd[mxrg - 1] - data_rd[nfubnd - 1] * data_rd[nfubnd - 1]);
    double sige1 = s0pv2 / 4.0;

    for(int lnuc = 0; lnuc < mxncfu; ++lnuc) {
        double sigt1 = 0.0;
        for (int ir = 0; ir < nfubnd; ++ir) {
            for (int inn = 0; inn < mnccel[ir]; ++inn) {
                if (inn == lnuc) continue;
                int id = data_idnc1[ir * 23 + inn];
                sigt1 += data_adnc[ir * 23 + inn] * data_volrg[ir] / volrgt * data_sigt[id];
            }
    }
        double sigt2 = 0.0;
        for (int ir = nfubnd; ir < mxrg; ++ir) {
            for (int inn = 0; inn < mnccel[ir]; ++inn) {
                int id = data_idnc1[ir * 23 + inn];
                sigt2 += data_adnc[ir * 23 + inn] * data_volrg[ir] / volrgt * data_sigt[id];
            }
        }

        double fgam = 1.0 / (1.0 + sige1 / sigt2);
        double fdanc = 1.0 - fgam - pow(fgam, 4) * (1.0 - fgam);

        double corec = 0.25 * fbell * (1.0 - fdanc) / (1.0 + (fbell - 1.0) * fdanc) * s0pv0 / data_atden[lnuc];
        data_sig0[lnuc] = sigt1 / data_atden[lnuc] + corec;
    }
}

__global__ void calsg1_kernel(
    int* data_idcl, double* data_rd, int nfubnd,
    int mxrg, int mxncfu, int* mnccel, int* data_idnc1, double* data_adnc, double* data_volrg,
    double volrgt, double* data_sigt, double fbell, double* data_atden, double* data_sig0,
    double* data_sig1d, int imax, int mclnuc, double temper, double* data_tab, double* data_x,
    double* data_g, double* data_gg, double* data_ftab, double* data_bg, double* data_cg, 
    double* data_dg, double* data_vt, double* data_tt, double* data_ft, int* data_msf,
    int* data_idnuc1, int mxsig0, int maxdat, double* data_s, double* data_a, double* data_b,
    double* data_c, double* data_d
    ) {
    
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    int inn = blockIdx.y * blockDim.y + threadIdx.y;

    if (ig < imax && inn < mclnuc) {
        data_sig0[inn] = 0.0;
        int nreac = 3;
        for (int i = 0; i < mclnuc; ++i) {
            if (i == inn) continue;
            int id = data_idcl[i];
            double tpsigt = data_sig1d[id * 8 * 70 + 0 * 70 + ig] + data_sig1d[id * 8 * 70 + 2 * 70 + ig] +
                            data_sig1d[id * 8 * 70 + 3 * 70 + ig] + data_sig1d[id * 8 * 70 + 4 * 70 + ig] +
                            data_sig1d[id * 8 * 70 + 7 * 70 + ig];
            double tpsig0 = 1.0e5;
            double xval = inter1(ig, nreac, id, tpsig0, temper, mxsig0, data_x, data_tab, data_idnuc1,
                                data_msf, data_g, data_ftab, data_gg, maxdat, data_s, data_a, data_b,
                                data_c, data_d, data_bg, data_cg, data_dg, data_vt, data_tt, data_ft);
            data_sig0[inn] += data_atden[i] * xval * tpsigt / data_atden[inn];
        }

        int id = data_idcl[inn];
        double xval = inter1(ig, nreac, id, data_sig0[inn], temper, mxsig0, data_x, data_tab, data_idnuc1,
                                data_msf, data_g, data_ftab, data_gg, maxdat, data_s, data_a, data_b,
                                data_c, data_d, data_bg, data_cg, data_dg, data_vt, data_tt, data_ft);
        double tpsigt = data_sig1d[id * 8 * 70 + 0 * 70 + ig] + data_sig1d[id * 8 * 70 + 2 * 70 + ig] +
                        data_sig1d[id * 8 * 70 + 3 * 70 + ig] + data_sig1d[id * 8 * 70 + 4 * 70 + ig] +
                        data_sig1d[id * 8 * 70 + 7 * 70 + ig];
        data_sigt[inn] = tpsigt * xval;

        for (int lp1 = 0; lp1 < 3; ++lp1) {
            // __syncthreads();
            calsg2(data_rd, nfubnd, mxrg, mxncfu, mnccel, data_idnc1, data_adnc, data_volrg, volrgt,
                    data_sigt, fbell, data_atden, data_sig0);
            // __syncthreads();
            xval = inter1(ig, nreac, id, data_sig0[inn], temper, mxsig0, data_x, data_tab, data_idnuc1,
                                data_msf, data_g, data_ftab, data_gg, maxdat, data_s, data_a, data_b,
                                data_c, data_d, data_bg, data_cg, data_dg, data_vt, data_tt, data_ft);
            tpsigt = data_sig1d[id * 8 * 70 + 0 * 70 + ig] + data_sig1d[id * 8 * 70 + 2 * 70 + ig] +
                    data_sig1d[id * 8 * 70 + 3 * 70 + ig] + data_sig1d[id * 8 * 70 + 4 * 70 + ig] +
                    data_sig1d[id * 8 * 70 + 7 * 70 + ig];
            data_sigt[inn] = tpsigt * xval;
        }
    }
}

__global__ void homog_kernel(
    int* data_idcl, double* data_sigt, double* data_sig0,
    double* data_sig0g, double* data_sigtg, double* data_sigf, double* data_sigc, double* data_sigel,
    double* data_siger, double* data_sigin, double* data_sig2, double* data_xnu, double* data_xmyu,
    double* data_siga, double* data_sig1d, int imax, int mclnuc, double temper, double* data_tab,
    double* data_x, double* data_g, double* data_gg, double* data_ftab, double* data_bg, double* data_cg,
    double* data_dg, double* data_vt, double* data_tt, double* data_ft, int* data_msf, int* data_idnuc1,
    int mxsig0, int maxdat, double* data_s, double* data_a, double* data_b, double* data_c, double* data_d
    ) {
    
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    int inn = blockIdx.y * blockDim.y + threadIdx.y;

    if (ig < imax && inn < mclnuc) {
        int idsp = data_idcl[inn];
        data_sig0g[inn * 70 + ig] = data_sig0[inn];
        data_sigtg[inn * 70 + ig] = data_sigt[inn];

        int nreac = 0;
        double fval = inter1(ig, nreac, idsp, data_sig0g[inn * 70 + ig], temper, mxsig0, data_x, data_tab, data_idnuc1,
                            data_msf, data_g, data_ftab, data_gg, maxdat, data_s, data_a, data_b,
                            data_c, data_d, data_bg, data_cg, data_dg, data_vt, data_tt, data_ft);
        data_sigf[inn * 70 + ig] = fval * data_sig1d[idsp * 8 * 70 + (1 - 1) * 70 + ig];

        nreac = 1;
        fval = inter1(ig, nreac, idsp, data_sig0g[inn * 70 + ig], temper, mxsig0, data_x, data_tab, data_idnuc1,
                            data_msf, data_g, data_ftab, data_gg, maxdat, data_s, data_a, data_b,
                            data_c, data_d, data_bg, data_cg, data_dg, data_vt, data_tt, data_ft);
        data_sigc[inn * 70 + ig] = fval * data_sig1d[idsp * 8 * 70 + (3 - 1) * 70 + ig];

        nreac = 2;
        fval = inter1(ig, nreac, idsp, data_sig0g[inn * 70 + ig], temper, mxsig0, data_x, data_tab, data_idnuc1,
                            data_msf, data_g, data_ftab, data_gg, maxdat, data_s, data_a, data_b,
                            data_c, data_d, data_bg, data_cg, data_dg, data_vt, data_tt, data_ft);
        data_sigel[inn * 70 + ig] = fval * data_sig1d[idsp * 8 * 70 + (5 - 1) * 70 + ig];

        nreac = 4;
        fval = inter1(ig, nreac, idsp, data_sig0g[inn * 70 + ig], temper, mxsig0, data_x, data_tab, data_idnuc1,
                            data_msf, data_g, data_ftab, data_gg, maxdat, data_s, data_a, data_b,
                            data_c, data_d, data_bg, data_cg, data_dg, data_vt, data_tt, data_ft);
        data_siger[inn * 70 + ig] = fval * data_sig1d[idsp * 8 * 70 + (7 - 1) * 70 + ig];

        nreac = 5;
        fval = inter1(ig, nreac, idsp, data_sig0g[inn * 70 + ig], temper, mxsig0, data_x, data_tab, data_idnuc1,
                            data_msf, data_g, data_ftab, data_gg, maxdat, data_s, data_a, data_b,
                            data_c, data_d, data_bg, data_cg, data_dg, data_vt, data_tt, data_ft);
        data_sigin[inn * 70 + ig] = fval * data_sig1d[idsp * 8 * 70 + (4 - 1) * 70+ ig];

        data_sig2[inn * 70 + ig] = data_sig1d[idsp * 8 * 70 + (8 - 1) * 70 + ig];
        data_xnu[inn * 70 + ig] = data_sig1d[idsp * 8 * 70 + (2 - 1) * 70 + ig];
        data_xmyu[inn * 70 + ig] = data_sig1d[idsp * 8 * 70 + (6 - 1) * 70 + ig];

        data_siga[inn * 70 + ig] = data_sigc[inn * 70 + ig] + data_sigf[inn * 70 + ig];
    }
}

__device__ double xlmdk(const double* data_rrg, const double* data_sigtrg, int ireg, int ig, double rho) {
    double delr = data_rrg[ireg] * data_rrg[ireg] - rho * rho;
    double delr1 = 0.0;
    if (ireg > 0) {
        delr1 = data_rrg[ireg - 1] * data_rrg[ireg - 1] - rho * rho;
    }

    double temp1 = (delr1 > 0.0) ? sqrt(delr1) : 0.0;
    double tmp = (delr > 0.0) ? sqrt(delr) : 0.0;

    return (tmp - temp1) * data_sigtrg[ireg * 70 + ig];
}

__device__ double xlmd1(const double* data_rrg, const double* data_sigtrg, int ii, int jj, int ig, double rho) {
    double tmp = 0.0;
    if (ii > jj) {
        for (int i = jj + 1; i < ii; ++i) {
            tmp += xlmdk(data_rrg, data_sigtrg, i, ig, rho);
        }
    } else {
        for (int i = ii + 1; i < jj; ++i) {
            tmp += xlmdk(data_rrg, data_sigtrg, i, ig, rho);
        }
    }
    return tmp;
}

__device__ double xlmd2(const double* data_rrg, const double* data_sigtrg, int ii, int jj, int ig, double rho) {
    double tmp = 0.0;
    for (int j = 0; j < jj; ++j) {
        tmp += xlmdk(data_rrg, data_sigtrg, j, ig, rho);
    }
    for (int i = 0; i < ii; ++i) {
        tmp += xlmdk(data_rrg, data_sigtrg, i, ig, rho);
    }
    return tmp;
}

__device__ double xlmd3(const double* data_rrg, const double* data_sigtrg, int ii, int ig, double rho) {
    double tmp = 0.0;
    if (ii != -1) {
        for (int j = 0; j < ii; ++j) {
            tmp += xlmdk(data_rrg, data_sigtrg, j, ig, rho);
        }
    }
    return tmp * 2.0;
}                    

__device__ double xlmds1(const double* data_rrg, const double* data_sigtrg, int ii, int ig, double rho, int maxcrg) {
    double tmp = 0.0;
    for (int j = ii + 1; j < maxcrg; ++j) {
        tmp += xlmdk(data_rrg, data_sigtrg, j, ig, rho);
    }
    return tmp;
}

__device__ double xlmds2(const double* data_rrg, const double* data_sigtrg, int ii, int ig, double rho, int maxcrg) {
    double tmp = 0.0;
    for (int j = 0; j < maxcrg; ++j) {
        tmp += xlmdk(data_rrg, data_sigtrg, j, ig, rho);
    }
    for (int i = 0; i < ii; ++i) {
        tmp += xlmdk(data_rrg, data_sigtrg, i, ig, rho);
    }
    return tmp;
}

__device__ double xki3as(double x) {
    double tmp1 = exp(-x) * sqrt(3.14159 / 2.0 / x);
    double bb = -13.0 / 8.0;
    double cc = 3.0 * (16.0 * 9.0 + 72.0 + 3.0) / 2.0 / 64.0;
    double dd = -2.5 * (64.0 * 27.0 + 240.0 * 9.0 + 212.0 * 3.0 + 15.0) / 512.0;
    double tmp2 = 1.0 + bb / x + cc / (x * x) + dd / (x * x * x);
    return tmp1 * tmp2;
}

__device__ double spfact(int n, int j) {
    double sptmp = 1.0;
    for (int i = 1; i <= n; ++i) {
        sptmp *= i / 2.0;
        if (i <= j) {
            sptmp /= i;
        }
        if (i <= n - j) {
            sptmp /= i;
        }
    }
    return sptmp;
}

__device__ double xki3s(double x) {
    double alfa = 5.0;
    int nmax = 10;
    double xkitmp = 0.0;
    double factj = 1.0;
    double tpsp1 = 1.0;
    double expalf = exp(alfa);
    
    xkitmp = 0.5 * sinh(alfa) / cosh(alfa) / cosh(alfa) + atan(expalf) - 3.14159 / 4.0;
    xkitmp -= x * tanh(alfa);
    xkitmp += x * x * (atan(expalf) - 3.14159 / 4.0) - x * x * x * alfa / 6.0;
    
    double sign = -1.0;
    double fact = 6.0;
    double xpn = x * x * x;
    
    for (int n = 1; n <= nmax; ++n) {
        fact *= (3 + n);
        xpn *= x;
        int np2 = n / 2;
        double tmp = xpn / fact * sign;
        
        if (n != 2 * np2) {
            double tmpint = 0.0;
            for (int j = 0; j <= n; ++j) {
                double tmpexp = alfa * (n - 2 * j);
                tmpint += spfact(n, j) * (exp(tmpexp) - 1.0) / (n - 2 * j);
            }
            xkitmp += tmp * tmpint;
        } else {
            double tmpint = 0.0;
            for (int j = 0; j < np2; ++j) {
                double tmpexp = alfa * (n - 2 * j);
                tmpint += spfact(n, j) * (exp(tmpexp) - exp(-tmpexp)) / (n - 2 * j);
            }
            tmpint += spfact(n, np2) * alfa;
            xkitmp += tmp * tmpint;
        }
        
        sign = -1.0 * sign;
    }
    return xkitmp;
}

__device__ double xki3(double x, int n) {
    double spl = 0.0;
    double xd2 = x / 2.0;
    double factj = 1.0;
    double tpsp1 = 1.0;

    if (x > 8.0) {
        return xki3as(x);
    }

    if (x < 0.05) {
        return xki3s(x);
    }

    double gamma = 0.577215664901532;
    double pi = 3.141592653589793;

    double vxki3 = 1.0 / 4.0 * pi - x + 0.25 * pi * x * x - (11.0 / 6.0 - gamma) / 6.0 * x * x * x + x * x * x / 6.0 * log(xd2);

    for (int j = 1; j <= n; ++j) {
        int j2p3 = 2 * j + 3;
        int j2p2 = 2 * j + 2;
        int j2p1 = 2 * j + 1;

        spl += 1.0 / j;
        tpsp1 *= xd2 / j;
        double tpsp2 = tpsp1 * tpsp1 * xd2 * xd2 * xd2 / (j2p3 * j2p2 * j2p1);
        vxki3 += 8.0 * log(xd2) * tpsp2;
        vxki3 += (gamma - 1.0 / j2p1 - 1.0 / j2p2 - 1.0 / j2p3 - spl) * 8.0 * tpsp2;
    }
    return vxki3;
}

__device__ double calcp0(const double* data_rrg, const double* data_absg, const double* data_weigg, const double* data_sigtrg, const double* data_volrrg, int ii, int iord, int ig) {
    int nbkf = 20;
    double xinteg = 0.0;
    if (ii > 0) {
        double rrm = data_rrg[ii - 1];

        for (int iteta = 0; iteta < iord; ++iteta) {
            double rho = rrm / 2.0 * (data_absg[iord * 100 + iteta] + 1.0);
            double xlamda = xlmdk(data_rrg, data_sigtrg, ii, ig, rho);
            double xlamd1 = xlamda; 
            xinteg += 2.0 * xlamda * data_weigg[iord * 100 + iteta]; 
            xinteg += 2.0 * data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);
            xlamda = 0.0;
            xinteg -= data_weigg[iord * 100 + iteta] * 3.141592653589793 / 2.0; 
            double xlamd3 = xlmd3(data_rrg, data_sigtrg, ii, ig, rho);
            xinteg += data_weigg[iord * 100 + iteta] * xki3(xlamd3, nbkf);
            xlamda = xlamd3 + xlamd1;
            xinteg -= 2.0 * data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);
            xlamda = xlamd3 + 2.0 * xlamd1;
            xinteg += data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);
        }

        xinteg = xinteg * 2.0 / data_sigtrg[ii * 70 + ig] / data_volrrg[ii] * rrm / 2.0;
    } else {
            double rrm = 0.0;}

    double rim, ri;
    if (ii == 0) {
        rim = 0.0;
        ri = data_rrg[0];
    } else {
        rim = data_rrg[ii - 1];
        ri = data_rrg[ii];
    }
                            
    double rav = (ri + rim) / 2.0;
    double delr = (ri - rim) / 2.0;
    double xsupl = 0.0;

    for (int iteta = 0; iteta < iord; ++iteta) {
        double rho = delr * (data_absg[iord * 100 + iteta] + 1.0) + rim;
        double xlamd1 = data_sigtrg[ii * 70 + ig] * sqrt(data_rrg[ii] * data_rrg[ii] - rho * rho);
        double xlamda = xlamd1;
        xsupl += 4.0 * xlamda * data_weigg[iord * 100 + iteta];
        xlamda = 2.0 * xlamd1;
        xsupl += 2.0 * data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);
        xlamda = 0.0;
        xsupl -= data_weigg[iord * 100 + iteta] * 3.141592653589793 / 2.0;
    }

    xsupl = xsupl / data_sigtrg[ii * 70 + ig] / data_volrrg[ii] * delr;
    return xinteg + xsupl;
}
                            
__global__ void calcp_kernel(double* pij, const double* data_rrg, const double* data_absg, const double* data_weigg, const double* data_sigtrg, const double* data_volrrg, int iord, int maxcrg, int imax, int nbkf) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    int ii = blockIdx.y * blockDim.y + threadIdx.y;
    int jj = blockIdx.z * blockDim.z + threadIdx.z;

    if (ig < imax && ii < maxcrg && jj < maxcrg) {
        if (ii == jj) {
            pij[ii * maxcrg * imax + jj * imax + ig] = calcp0(data_rrg, data_absg, data_weigg, data_sigtrg, data_volrrg, ii, iord, ig);
        } else {
            double xinteg = 0.0;
            for (int iteta = 0; iteta < iord; ++iteta) {
                double rho = data_rrg[ii] / 2.0 * (data_absg[iord * 100 + iteta] + 1.0);
                double xlamda = xlmd1(data_rrg, data_sigtrg, ii, jj, ig, rho);
                double xlamd1 = xlamda;
                xinteg += data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);
                xlamda = xlamd1 + xlmdk(data_rrg, data_sigtrg, ii, ig, rho);
                xinteg -= data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);
                xlamda = xlamd1 + xlmdk(data_rrg, data_sigtrg, jj, ig, rho);
                xinteg -= data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);
                xlamda = xlamd1 + xlmdk(data_rrg, data_sigtrg, ii, ig, rho) + xlmdk(data_rrg, data_sigtrg, jj, ig, rho);
                xinteg += data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);

                xlamda = xlmd2(data_rrg, data_sigtrg, ii, jj, ig, rho);
                double xlamd2 = xlamda;
                xinteg += data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);
                xlamda = xlamd2 + xlmdk(data_rrg, data_sigtrg, ii, ig, rho);
                xinteg -= data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);
                xlamda = xlamd2 + xlmdk(data_rrg, data_sigtrg, jj, ig, rho);
                xinteg -= data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);
                xlamda = xlamd2 + xlmdk(data_rrg, data_sigtrg, ii, ig, rho) + xlmdk(data_rrg, data_sigtrg, jj, ig, rho);
                xinteg += data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);
            }

            double valcp = xinteg * 2.0 / data_sigtrg[ii * 70 + ig] / data_volrrg[ii] * data_rrg[ii] / 2.0;
            pij[ii * maxcrg * imax + jj * imax + ig] = valcp;
        }
    }
}

__global__ void escpr_kernel(
    double* pesc, const double* data_rrg, const double* data_absg, const double* data_weigg, const double* data_sigtrg, const double* data_volrrg, int iord, int maxcrg, int imax, int nbkf
    ) {
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    int ii = blockIdx.y * blockDim.y + threadIdx.y;

    if (ii < maxcrg && ig < imax) {
        double xinteg = 0.0;
        for (int iteta = 0; iteta < iord; iteta++) {
            double rho = data_rrg[ii] / 2.0 * (data_absg[iord * 100 + iteta] + 1.0);
            double xlamd1 = xlmds1(data_rrg, data_sigtrg, ii, ig, rho, maxcrg);
            double xlamda = xlamd1;
            xinteg += data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);
            xlamda = xlamd1 + xlmdk(data_rrg, data_sigtrg, ii, ig, rho);
            xinteg -= data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);
            double xlamd2 = xlmds2(data_rrg, data_sigtrg, ii, ig, rho, maxcrg);
            xlamda = xlamd2;
            xinteg += data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);
            xlamda = xlamd2 + xlmdk(data_rrg, data_sigtrg, ii, ig, rho);
            xinteg -= data_weigg[iord * 100 + iteta] * xki3(xlamda, nbkf);
        }

        double escprb = xinteg * 2.0 / data_sigtrg[ii * 70 + ig] / data_volrrg[ii] * data_rrg[ii] / 2.0;
        pesc[ii * imax + ig] = escprb;
    }
}
}