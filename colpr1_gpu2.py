import numpy as np
import cupy as cp
import math
import time
from comclhmg import ComclhmgData

with open('D:\Tesis Ghulam Abrar\colpr1_kernel2.cu', 'r') as f:
    kernel_code = f.read()

# Compile the CUDA kernel code
module = cp.RawModule(code=kernel_code)

# Get the kernel function from the compiled module
calcp_kernel = module.get_function('calcp_kernel')
escpr_kernel = module.get_function('escpr_kernel')
calsg1_kernel = module.get_function('calsg1_kernel')
homog_kernel = module.get_function('homog_kernel')

# Create an instance of ComclhmgData
data = ComclhmgData()

def read1(filename, data: ComclhmgData):
    # with open(filename, 'r') as file:
    #     content = file.read()
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Create a new content string without comment lines
    content = ""
    for line in lines:
        if not line.strip().startswith('c'):
            content += line

    values = list(map(float, content.split()))

    header_values = values[:7]
    data.mreduc = int(header_values[0])
    data.temper = float(header_values[1])
    data.mclnuc = int(header_values[2])
    data.mxrg = int(header_values[3])
    data.nfubnd = int(header_values[4])
    data.fbell = int(header_values[5])
    data.nord = int(header_values[6])
    print(f'mreduc, temper, mclnuc, mxrg, nfubnd, fbell, nord')
    print(f'{data.mreduc},{data.temper}, {data.mclnuc}, {data.mxrg}, {data.nfubnd}, {data.fbell}, {data.nord}')

    data.idnuc = np.zeros(data.mreduc, dtype=int)
    data.rd = np.zeros(data.mxrg, dtype=float)

    index = 7
    idnuc_values = values[index:index+data.mreduc]
    index += data.mreduc
    rd_values = values[index:index+data.mxrg]
    index += data.mxrg

    for i in range(data.mreduc): data.idnuc[i] = idnuc_values[i]
    for i in range(data.mxrg): data.rd[i] = rd_values[i]

    header2_values = values[index:index+4]
    index += 4
    data.xkef0 = float(header2_values[0])
    data.errfl = float(header2_values[1])
    data.errkef = float(header2_values[2])
    data.nchi = int(header2_values[3])
    print(f'xkef0, errfl, errkef, nchi')
    print(f'{data.xkef0},{data.errfl}, {data.errkef}, {data.nchi}')

    data.mnccel = np.zeros(data.mxrg, dtype=int)

    mnccel_values = values[index:index+data.mxrg]
    index += data.mxrg

    for i in range(data.mxrg): data.mnccel[i] = mnccel_values[i]

    itp = 0
    maxtp = 0
    maxtp1 = 0
    volrgt = 0.0
    print(f'(idnc(i,j),adnc(i,j),j=1,mnccel(i))')
    data.idnc = np.zeros((data.mxrg, np.max(data.mnccel)), dtype=int)
    data.adnc = np.zeros((data.mxrg, np.max(data.mnccel)), dtype=float)
    data.idnc1 = np.zeros((data.mxrg, np.max(data.mnccel)), dtype=int)
    
    for i in range(data.mxrg):
        for j in range(data.mnccel[i]):
            idnc_values = values[index]
            index += 1
            adnc_values = values[index]
            index += 1
            data.idnc[i, j] = int(idnc_values)
            data.adnc[i, j] = float(adnc_values)
        
        if i == 0:
            data.volrg[0] = 3.14159 * data.rd[0] ** 2
        else:
            data.volrg[i] = 3.14159 * (data.rd[i] ** 2 - data.rd[i - 1] ** 2)
        
        data.volrgt += data.volrg[i]

        for j in range(data.mnccel[i]):
            itp += 1

            found = False
            for i1 in range(maxtp + 1):
                if data.ncod[i1] == data.idnc[i, j]:
                    data.idnc1[i, j] = i1
                    found = True
                    continue

                if not found:
                    data.ncod[maxtp1] = data.idnc[i, j]
                    data.idnc1[i, j] = maxtp1
                    maxtp1 = maxtp + 1

                # print(f'idnc1: {i}, {j}, {i1}, {itp}, {maxtp}')
            maxtp = maxtp1
        
        # print(f'idnc1 = {i}', [f"{data.idnc1[i, j1]}" for j1 in range(data.mnccel[i])])

    data.mcpreg = np.zeros(data.mxrg, dtype=int)
    mcpreg_values = values[index:index+data.mxrg]
    index += data.mxrg
    for i in range (data.mxrg): data.mcpreg[i] = mcpreg_values[i]
    print('(mcpreg(i), i=0, mxrg)')
    print([f"{data.mcpreg[i]}" for i in range(data.mxrg)])

    data.mclnuc = maxtp
    data.mxncfu = maxtp

    print(f'volrgt={data.volrgt}')
    print([f"{data.volrg[i]}" for i in range(data.mxrg)])
    print('ncod(i):')
    print([f"{data.ncod[i]}" for i in range(maxtp)])

    for i in range(data.mclnuc):
        data.atden[i] = 0.0
        data.anfu[i] = 0.0

    for i in range(data.mxrg):
        for j in range(data.mnccel[i]):
            id = data.idnc1[i, j]
            data.atden[id] += data.adnc[i, j] * data.volrg[i] / data.volrgt
            if i > data.nfubnd - 1:
                continue
            data.anfu[id] += data.adnc[i, j] * data.volrg[i] / data.volrgt
    
    print(f"atden(i): {data.mclnuc}")
    print([f"{data.atden[i]:.2f}" for i in range(data.mclnuc)])

    # for i in range(0, 80): #Kenapa 80? kenapa bukan data.lnmax?
    data.ncstat = np.zeros(391, dtype=int)
    data.idnuc1 = np.zeros(391, dtype=int)
    for i in range(0, 391): #391 adalah lnmax dari libcel.dat
        data.ncstat[i] = 0
        data.idnuc1[i] = 0
        for j in range(data.mreduc):
            if i+1 == data.idnuc[j]:
                data.ncstat[i] = 1
                data.idnuc1[i] = j
        # print(f"{i}, {data.ncstat[i]}, {data.idnuc1[i]}")

def suppl1(data: ComclhmgData, ng, mode):
    name1, name2 = "", ""  # Character variables in Fortran

    for i in range(data.lnmax):
        if data.ncstat[i] != 1:
            continue

        if mode == 1:
            max1 = data.mxdnse
        elif mode == 2:
            max1 = data.mxdnsi
        elif mode == 3:
            max1 = data.mxdns2

        itp = data.idnuc1[i]

        for j in range(max1):
            igj = ng + j #check, looks good
            if igj > data.imax:
                continue

            if mode == 1:
                data.stre[ng, itp, igj] = data.tmp2[j, i]
            elif mode == 2:
                data.stri[ng, itp, igj] = data.tmp2[j, i]
            elif mode == 3:
                data.str2[ng, itp, igj] = data.tmp2[j, i]

def supl2(data: ComclhmgData, ng, kt, mr, m, mt):
    name1, name2 = "", ""  # Character variables in Fortran

    itp = data.idnuc1[m]
    if itp == -1: #check again
        return

    for j in range(data.mxsig0):
        for k in range(kt):
            data.ftab[j, k, ng, mt, itp] = data.tmp1[j, k, 0]

def rdlbsp(filename, data: ComclhmgData):
    with open(filename, 'r') as file:
        content = file.read()
    
    values = list(map(float, content.split()))

    print("Reading header")

    header_values = list(map(int, values[:16]))
    data.lnmax, data.imax, data.mxchi, data.mxr1d, data.mtxr23, data.mxdnse, \
    data.mxdnsi, data.mxdns2, data.mxdwne, data.mxdwni, data.mxdwn2, data.mxreac, \
    data.mxsig0, data.mxtemp, data.mxr, data.iswh = header_values

    print(f"lnmax, imax, mxchi, mxr1d, mtxr23, mxdnse, mxdnsi, mxdns2, "
          f"mxdwne, mxdwni, mxdwne, mxreac, mxsig0, mxtemp, mxr, iswh")
    print(f"{data.lnmax}, {data.imax}, {data.mxchi}, {data.mxr1d}, {data.mtxr23}, {data.mxdnse}, "
          f"{data.mxdnsi}, {data.mxdns2}, {data.mxdwne}, {data.mxdwni}, {data.mxdwne}, {data.mxreac}, "
          f"{data.mxsig0}, {data.mxtemp}, {data.mxr}, {data.iswh}")
    
    print(f'reading ncode ')
    data.ncodel = np.zeros(data.lnmax, dtype=int)
    data.enbnd = np.zeros(data.imax + 1, dtype=float)
    data.chi = np.zeros((data.mxchi, data.imax), dtype=float)
    data.ncodx = np.zeros(data.mxchi, dtype=int)
    data.aw = np.zeros(data.lnmax, dtype=float)
    data.ld = np.zeros((data.lnmax, 3), dtype=int)
    data.la = np.zeros((data.lnmax, 3), dtype=int)
    data.msf = np.zeros((data.lnmax, data.mxreac), dtype=int)
    data.tab = np.zeros((data.lnmax, data.mxsig0), dtype=float)
    data.ft = np.zeros((data.lnmax, data.mxtemp), dtype=float)
    data.rparam = np.zeros((data.lnmax, data.mxr, data.mxsig0), dtype=float)
    data.ntemp = np.zeros((data.lnmax, data.mxreac), dtype=int)
    data.nr = np.zeros((data.lnmax, data.mxreac), dtype=int)

    index = 16
    ncodel_values = values[index:index+data.lnmax]
    index += data.lnmax
    enbnd_values = values[index:index+data.imax + 1]
    index += data.imax + 1
    chi_values = values[index:index + data.imax * data.mxchi]
    index += data.imax * data.mxchi
    ncodx_values = values[index:index + data.mxchi]
    index += data.mxchi
    aw_values = values[index:index + data.lnmax]
    index += data.lnmax
    ld_values = values[index:index + 3 * data.lnmax]
    index += 3 * data.lnmax
    la_values = values[index:index + 3 * data.lnmax]
    index += 3 * data.lnmax
    msf_values = values[index:index + data.mxreac * data.lnmax]
    index += data.mxreac * data.lnmax
    tab_values = values[index:index + data.mxsig0 * data.lnmax]
    index += data.mxsig0 * data.lnmax
    ft_values = values[index:index + data.mxtemp * data.lnmax]
    index += data.mxtemp * data.lnmax
    rparam_values = values[index:index + data.mxsig0 * data.mxr * data.lnmax]
    index += data.mxsig0 * data.mxr * data.lnmax
    ntemp_values = values[index:index + data.mxreac * data.lnmax]
    index += data.mxreac * data.lnmax
    nr_values = values[index:index + data.mxreac * data.lnmax]
    index += data.mxreac * data.lnmax

    for i in range(data.lnmax): data.ncodel[i] = ncodel_values[i]
    for i in range(data.imax + 1): data.enbnd[i] = enbnd_values[i]
    for i in range(data.mxchi):
        for j in range(data.imax):
            data.chi[i, j] = chi_values[i * data.imax + j]
    data.chi = data.chi.T
    for i in range(data.mxchi): data.ncodx[i] = ncodx_values[i]
    for i in range(data.lnmax): data.aw[i] = aw_values[i]
    for i in range(data.lnmax):
        for j in range(3):
            data.ld[i, j] = ld_values[i * 3 + j]
    data.ld = data.ld.T
    for i in range(data.lnmax):
        for j in range(3):
            data.la[i, j] = la_values[i * 3 + j]
    data.la = data.la.T
    for i in range(data.lnmax):
        for j in range(data.mxreac):
            data.msf[i, j] = msf_values[i * data.mxreac + j]
    data.msf = data.msf.T
    for i in range(data.lnmax):
        for j in range(data.mxsig0):
            data.tab[i, j] = tab_values[i * data.mxsig0 + j]
    data.tab = data.tab.T
    for i in range(data.lnmax):
        for j in range(data.mxtemp):
            data.ft[i, j] = ft_values[i * data.mxtemp + j]
    data.ft = data.ft.T
    for i in range(data.lnmax):
        for j in range(data.mxr):
            for k in range(data.mxsig0):
                data.rparam[i, j, k] = rparam_values[i * data.mxr * data.mxsig0 + j * data.mxsig0 + k]
    data.rparam = np.transpose(data.rparam, (2, 1, 0))
    for i in range(data.lnmax):
        for j in range(data.mxreac):
            data.ntemp[i, j] = ntemp_values[i * data.mxreac + j]
    data.ntemp = data.ntemp.T
    for i in range(data.lnmax):
        for j in range(data.mxreac):
            data.nr[i, j] = nr_values[i * data.mxreac + j]
    data.nr = data.nr.T
    
    data.sig1d = np.zeros((data.lnmax, data.mxr1d, data.imax), dtype=float)
    sig1d_array = []
    data.stre = np.zeros((data.imax, 80, data.imax+data.mxdnse), dtype=float)
    data.stri = np.zeros((data.imax, 80, data.imax+data.mxdnsi), dtype=float)
    data.str2 = np.zeros((data.imax, 80, data.imax+data.mxdns2), dtype=float)
    data.ftab = np.zeros((data.mxsig0, np.max(data.msf), data.imax, data.mxreac, data.mreduc))
    
    for i in range(data.imax):
        sig1d_values = values[index:index + data.lnmax * data.mxr1d]
        index += data.lnmax * data.mxr1d
        sig1d_temp = np.zeros((data.mxr1d, data.lnmax), dtype=float)
        for i1 in range(data.mxr1d):
            for j1 in range(data.lnmax):
                sig1d_temp[i1, j1] = sig1d_values[i1 * data.lnmax + j1]
        sig1d_temp = sig1d_temp.T
        sig1d_array.append(sig1d_temp)

        if i <= data.mxdwne:
            tmp2_values = values[index:index + data.mxdnse * data.lnmax]
            index += data.mxdnse * data.lnmax
            data.tmp2 = np.zeros((data.lnmax, data.mxdnse), dtype=float)
            for j in range(data.lnmax):
                for m in range(data.mxdnse):
                    data.tmp2[j, m] = tmp2_values[j * data.mxdnse + m]
            data.tmp2 = data.tmp2.T
        
        mode = 1
        suppl1(data, i, mode)

        if data.iswh != 0:
            itp = data.idnuc1[data.iswh - 1]
            if itp != 0: #check again
                stre_values = values[index:index + data.imax]
                index += data.imax
                for j in range(data.imax): 
                    data.stre[j, itp, i] = stre_values[j]
   
            else:
                tmp2_values = values[index:index + int(data.imax)]
                index += data.imax
                data.tmp2 = np.pad(data.tmp2, ((0, data.imax-data.mxdnse), (0, 0)), 'constant', constant_values=0)
                for j in range(data.imax):
                    data.tmp2[j, itp] = tmp2_values[j]
        if i < data.mxdwni:
            tmp2_values = values[index:index + data.mxdnsi * data.lnmax]
            index += data.mxdnsi * data.lnmax
            data.tmp2 = np.zeros((data.lnmax, data.mxdnsi), dtype=float)
            for j in range(data.lnmax):
                for m in range(data.mxdnsi):
                    data.tmp2[j, m] = tmp2_values[j * data.mxdnsi + m]
            data.tmp2 = data.tmp2.T
        
        mode = 2
        suppl1(data, i, mode)

        if i < data.mxdwn2:
            tmp2_values = values[index:index + data.mxdns2 * data.lnmax]
            index += data.mxdns2 * data.lnmax
            data.tmp2 = np.zeros((data.lnmax, data.mxdns2), dtype=float)
            for j in range(data.lnmax):
                for m in range(data.mxdns2):
                    data.tmp2[j, m] = tmp2_values[j * data.mxdns2 + m]
            data.tmp2 = data.tmp2.T
        
        mode = 3
        suppl1(data, i, mode)

        for m in range(data.lnmax):
            for mt in range(data.mxreac):
                mr = data.nr[mt, m]
                kt = data.msf[mt, m]
                itp = data.idnuc1[m]

                if mr == 0 or kt == 0:
                    continue
                
                if mt != 5:
                    tmp1_values = values[index:index + data.mxsig0 * kt * mr]
                    index += data.mxsig0 * kt * mr
                    data.tmp1 = np.zeros((data.mxsig0, kt, mr), dtype=float)
                    for j in range(data.mxsig0):
                        for k in range(kt):
                            for n in range(mr):
                                data.tmp1[j, k, n] = tmp1_values[j * kt * mr + k * mr + n]
                    ng = i
                    supl2(data, ng, kt, mr, m, mt)

                if mt == 5 and data.la[1, m] >= i+1:
                    tmp1_values = values[index:index + data.mxsig0 * kt * mr]
                    index += data.mxsig0 * kt * mr
                    data.tmp1 = np.zeros((data.mxsig0, kt, mr), dtype=float)
                    for j in range(data.mxsig0):
                        for k in range(kt):
                            for n in range(mr):
                                data.tmp1[j, k, n] = tmp1_values[j * kt * mr + k * mr + n]
                    ng = i
                    supl2(data, ng, kt, mr, m, mt)
    data.sig1d = np.stack(sig1d_array, axis=2)

def scelas(data: ComclhmgData, ncl):
    irec = data.idcl1[ncl]
    nuc = data.idcl[ncl]
    aa = data.aw[nuc]
    # print(f'nucleid number :ncl,irec,nuc {ncl}, {irec}, {nuc}')
    alfa = ((aa - 1.0) / (aa + 1.0)) ** 2
    # print(f'atomic weight of the nucleus : {aa}')
    # print(f'Alfa=((A-1)/(A+1))**2 = {alfa}')

    for igs in range(data.imax):
        esup = data.enbnd[igs]
        eslow = data.enbnd[igs + 1]
        for igt in range(data.imax):
            icase = 6
            etup = data.enbnd[igt]
            etlow = data.enbnd[igt + 1]
            if igs > igt:
                icase = 6 #goto 25
            elif igt == igs:
                icase = 0 #goto 25
            elif alfa * esup <= etlow:
                icase = 1 #goto 25
            elif (alfa * esup <= etup) and (alfa * eslow <= etlow):
                icase = 2 
            elif (alfa * esup >= etup) and (alfa * eslow <= etlow):
                icase = 3
            elif (alfa * esup <= etup) and (alfa * eslow <= etup):
                icase = 4
            elif (alfa * esup >= etup) and (alfa * eslow < etup):
                icase = 5
            else:
                icase = 6

            sgstp = 0.0 #25
            if icase == 0:
                if alfa * esup < eslow:
                    sgstp = (esup - eslow) / (1.0 - alfa) * np.log(1.0 / alfa)
                else:
                    sgstp = (esup - eslow) / (1.0 - alfa) - eslow / (1.0 - alfa) * np.log(esup / eslow)
            elif icase == 1:
                sgstp = (etup - etlow) / (1.0 - alfa) * np.log(esup / eslow)
            elif icase == 2:
                sgstp = ((etup - etlow) / (1.0 - alfa) * np.log(etlow / (alfa * eslow)) +
                         (etup - alfa * esup) / (1.0 - alfa) * np.log(esup * alfa / etlow) +
                         alfa * esup / (1.0 - alfa) * np.log(esup * alfa / etlow) -
                         alfa / (1.0 - alfa) * (esup - etlow / alfa))
            elif icase == 3:
                sgstp = ((etup - etlow) / (1.0 - alfa) * np.log(etlow / (alfa * eslow)) +
                         etup / (1.0 - alfa) * np.log(etup / etlow) -
                         (etup - etlow) / (1.0 - alfa))
            elif icase == 4:
                sgstp = ((etup - alfa * esup) / (1.0 - alfa) * np.log(esup / eslow) +
                         alfa * esup / (1.0 - alfa) * np.log(esup / eslow) -
                         (esup - eslow) * alfa / (1.0 - alfa))
            elif icase == 5:
                sgstp = (etup / (1.0 - alfa) * np.log(etup / (alfa * eslow)) -
                         alfa / (1.0 - alfa) * (etup / alfa - eslow))
            
            sgstp = sgstp / (esup - eslow)
            data.stre[igs, irec, igt] = sgstp * data.sigel[ncl, igs]

def homog(data: ComclhmgData):
    for i in range(data.mclnuc):
        match = 0
        for j in range(data.lnmax):
            if data.ncod[i] == data.ncodel[j]:
                data.idcl[i] = j

    for i in range(data.mclnuc):
        itp = data.idcl[i]
        data.idcl1[i] = data.idnuc1[itp]
        # print(f"idcl1: {i}, {data.idcl1[i]}")

    # print("\nlist of temperature")
    # for i in range(data.lnmax):
    #     print(f"{i}, {data.ncodel[i]}", [f"{data.ft[j, i]:.5f}" for j in range(data.mxtemp)])

    # print("\nlist of sigma0 value")
    # for i in range(data.lnmax):
    #     print(f"{i}", [f"{data.tab[j, i]:.5f}" for j in range(data.mxsig0)])
    
    # print(f'ig,  vsig0,  temper,  fval')

    data_rd = cp.asarray(data.rd, dtype=cp.float64)
    data_mnccel = cp.asarray(data.mnccel, dtype=cp.int32)
    data_idnc1 = cp.asarray(data.idnc1.flatten()).astype(cp.int32)
    data_adnc = cp.asarray(data.adnc.flatten()).astype(cp.float64)
    data_volrg = cp.asarray(data.volrg, dtype=cp.float64)
    data_atden = cp.asarray(data.atden, dtype=cp.float64)


    data_sig0 = cp.asarray(data.sig0, dtype=cp.float64)
    data_sigt = cp.asarray(data.sigt, dtype=cp.float64)
    data_sig0g = cp.asarray(data.sig0g.flatten()).astype(cp.float64)
    data_sigtg = cp.asarray(data.sigtg.flatten()).astype(cp.float64)
    data_sigf = cp.asarray(data.sigf.flatten()).astype(cp.float64)
    data_sigc = cp.asarray(data.sigc.flatten()).astype(cp.float64)
    data_sigel = cp.asarray(data.sigel.flatten()).astype(cp.float64)
    data_siger = cp.asarray(data.siger.flatten()).astype(cp.float64)
    data_sigin = cp.asarray(data.sigin.flatten()).astype(cp.float64)
    data_sig2 = cp.asarray(data.sig2.flatten()).astype(cp.float64)
    data_xnu = cp.asarray(data.xnu.flatten()).astype(cp.float64)
    data_xmyu = cp.asarray(data.xmyu.flatten()).astype(cp.float64)
    data_siga = cp.asarray(data.siga.flatten()).astype(cp.float64)

    data_idcl = cp.asarray(data.idcl, dtype=cp.int32)
    data_sig1d = cp.asarray(data.sig1d.flatten()).astype(cp.float64)
    data_tab = cp.asarray(data.tab.flatten()).astype(cp.float64)
    data_x = cp.asarray(data.x, dtype=cp.float64)
    data_g = cp.asarray(data.g, dtype=cp.float64)
    data_gg = cp.asarray(data.gg.flatten()).astype(cp.float64)
    data_ftab = cp.asarray(data.ftab.flatten()).astype(cp.float64)
    data_s = cp.asarray(data.s, dtype=cp.float64)
    data_a = cp.asarray(data.a.flatten()).astype(cp.float64)
    data_b = cp.asarray(data.b, dtype=cp.float64)
    data_c = cp.asarray(data.c, dtype=cp.float64)
    data_d = cp.asarray(data.d, dtype=cp.float64)
    data_bg = cp.asarray(data.bg.flatten()).astype(cp.float64)
    data_cg = cp.asarray(data.cg.flatten()).astype(cp.float64)
    data_dg = cp.asarray(data.dg.flatten()).astype(cp.float64)
    data_vt = cp.asarray(data.vt, dtype=cp.float64)
    data_tt = cp.asarray(data.tt, dtype=cp.float64)
    data_ft = cp.asarray(data.ft.flatten()).astype(cp.float64)
    data_msf = cp.asarray(data.msf.flatten()).astype(cp.int32)
    data_idnuc1 = cp.asarray(data.idnuc1, dtype=cp.int32)

    

    threads_per_block = (32, 16)
    blocks_per_grid = (
        (data.imax + threads_per_block[0] - 1) // threads_per_block[0],
        (data.mclnuc + threads_per_block[1] - 1) // threads_per_block[1]
    )

    # Launch the calsg1 kernel
    calsg1_kernel(
        blocks_per_grid, threads_per_block,
        (
            data_idcl, data_rd, data.nfubnd, data.mxrg,
            data.mxncfu, data_mnccel, data_idnc1, data_adnc, data_volrg, data.volrgt, data_sigt,
            data.fbell, data_atden, data_sig0, data_sig1d, data.imax, data.mclnuc, data.temper,
            data_tab, data_x, data_g, data_gg, data_ftab, data_bg, data_cg, data_dg,
            data_vt, data_tt, data_ft, data_msf, data_idnuc1, data.mxsig0, data.maxdat, 
            data_s, data_a, data_b, data_c, data_d
        )
    )
    # Launch the main kernel
    data.idcl = cp.asnumpy(data_idcl)
    homog_kernel(
        blocks_per_grid, threads_per_block,
        (
            data_idcl, data_sigt,
            data_sig0g, data_sigtg, data_sigf, data_sigc, data_sigel, data_siger, data_sigin,
            data_sig2, data_xnu, data_xmyu, data_siga, data_sig0, data_sig1d, data.imax,
            data.mclnuc, data.temper, data_tab, data_x, data_g, data_gg, data_ftab, data_bg,
            data_cg, data_dg, data_vt, data_tt, data_ft, data_msf, data_idnuc1, data.mxsig0,
            data.maxdat, data_s, data_a, data_b, data_c, data_d
        )
    )
    data.rd = cp.asnumpy(data_rd)
    data.mnccel = cp.asnumpy(data.mnccel)
    data.idnc1 = cp.asnumpy(data_idnc1).reshape(data.idnc1.shape)
    data.adnc = cp.asnumpy(data.adnc).reshape(data.adnc.shape)
    data.volrg = cp.asnumpy(data_volrg)
    data.atden = cp.asnumpy(data_atden)

    data.sig0 = cp.asnumpy(data_sig0)
    data.sigt = cp.asnumpy(data_sigt)
    data.sig0g = cp.asnumpy(data_sig0g).reshape(data.sig0g.shape)
    data.sigtg = cp.asnumpy(data_sigtg).reshape(data.sigtg.shape)
    data.sigf = cp.asnumpy(data_sigf).reshape(data.sigf.shape)
    data.sigc = cp.asnumpy(data_sigc).reshape(data.sigc.shape)
    data.sigel = cp.asnumpy(data_sigel).reshape(data.sigel.shape)
    data.siger = cp.asnumpy(data_siger).reshape(data.siger.shape)
    data.sigin = cp.asnumpy(data_sigin).reshape(data.sigin.shape)
    data.sig2 = cp.asnumpy(data_sig2).reshape(data.sig2.shape)
    data.xnu = cp.asnumpy(data.xnu).reshape(data.xnu.shape)
    data.xmyu = cp.asnumpy(data.xmyu).reshape(data.xmyu.shape)
    data.siga = cp.asnumpy(data.siga).reshape(data.siga.shape)

    data.idcl = cp.asnumpy(data_idcl)
    data.sig1d = cp.asnumpy(data_sig1d).reshape(data.sig1d.shape)
    data.tab = cp.asnumpy(data_tab).reshape(data.tab.shape)
    data.x = cp.asnumpy(data_x)
    data.g = cp.asnumpy(data_g)
    data.gg = cp.asnumpy(data_gg).reshape(data.gg.shape)
    data.ftab = cp.asnumpy(data_ftab).reshape(data.ftab.shape)
    data.s = cp.asnumpy(data_s)
    data.a = cp.asnumpy(data_a).reshape(data.a.shape)
    data.b = cp.asnumpy(data_b)
    data.c = cp.asnumpy(data_c)
    data.d = cp.asnumpy(data_d)
    data.bg = cp.asnumpy(data_bg).reshape(data.bg.shape)
    data.cg = cp.asnumpy(data_cg).reshape(data.cg.shape)
    data.dg = cp.asnumpy(data.dg).reshape(data.dg.shape)
    data.vt = cp.asnumpy(data_vt)
    data.tt = cp.asnumpy(data.tt)
    data.ft = cp.asnumpy(data_ft).reshape(data.ft.shape)
    data.msf = cp.asnumpy(data.msf).reshape(data.msf.shape)
    data.idnuc1 = cp.asnumpy(data_idnuc1)

    
    # for i in range(data.mclnuc):
    #     print(f"fission cross section for: {data.ncod[i]}")
    #     print([f"{data.sigf[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"capture cross section for: {data.ncod[i]}")
    #     print([f"{data.sigc[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"elastic cross section for: {data.ncod[i]}")
    #     print([f"{data.sigel[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"elastic removal cross section for: {data.ncod[i]}")
    #     print([f"{data.siger[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"inelastic cross section for: {data.ncod[i]}")
    #     print([f"{data.sigin[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"n2n cross section for: {data.ncod[i]}")
    #     print([f"{data.sig2[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"total cross section for: {data.ncod[i]}")
    #     print([f"{data.sigtg[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"absorbtion cross section for: {data.ncod[i]}")
    #     print([f"{data.siga[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"sig0 cross section for: {data.ncod[i]}")
    #     print([f"{data.sig0g[i, j]:.5f}" for j in range(data.imax)])

    for inn in range(data.mclnuc):
        scelas(data, inn)

def incoef(data: ComclhmgData):
    with open('D:\Tesis Ghulam Abrar\gausquad.dat', 'r') as file:  # Open the input file
        file.readline()
        file.readline()

        while True:
            line = file.readline().strip()
            if not line:
                break  # End of file

            if line.startswith('order of gauss integration'):
                order_line = file.readline().strip()
                norde = int(order_line)
                print(f'norde= {norde}')

                file.readline()  # Read and ignore the line with 'number   absis , weight'

                for inn in range(norde):
                    line = file.readline().strip()
                    parts = line.split()
                    idum = int(parts[0])
                    data.absg[norde-2, inn] = float(parts[1])
                    data.weigg[norde-2, inn] = float(parts[2])
                    # print(idum, data.absg[norde-2, inn], data.weigg[norde-2, inn])

def pijadj(data: ComclhmgData):
    print("\n----------pijadj() start----------")
    print("Collision probability after adjustment")
    surf = 2.0 * 3.14159 * data.rrg[data.maxcrg - 1]

    for ig in range(data.imax):
        data.q00[ig] = 1.0
        for irg in range(data.maxcrg):
            data.pescq[irg, ig] = (4.0 * data.volrrg[irg] / surf *
                                   data.sigtrg[irg, ig] * data.pesc[irg, ig])
            data.q00[ig] -= data.pescq[irg, ig]
            print(f"Q00: {irg}, {ig}, {data.pescq[irg, ig]:.5f}, {data.q00[ig]:.5f}")

        print(f"{ig:5}, {data.q00[ig]:.5f}",
              [f"{data.pescq[i3, ig]:.5f}" for i3 in range(data.maxcrg)])

    for ig in range(data.imax):
        for ii in range(data.maxcrg):
            for jj in range(data.maxcrg):
                data.pij[ii, jj, ig] += (data.pesc[ii, ig] * data.pescq[jj, ig] /
                                         (1.0 - data.q00[ig]))
    
    for ig in range(data.imax):
        print(f"Energy group: {ig}")
        print("Escape probability pesc(i), i=1, maxcrg")
        print([f"{data.pesc[i3, ig]:.5f}" for i3 in range(data.maxcrg)])
        print("Collision prob. pij i:! j ->")

        for ii in range(data.maxcrg):
            print(f"{ii:5}",
                  [f"{data.pij[ii, i3, ig]:.5f}" for i3 in range(data.maxcrg)])
    print("----------pijadj() end----------\n")

def colpr1(data: ComclhmgData):
    # print("\n----------colpr1() start----------")
    ii = 0
    rb = 0.0

    for ireg in range(data.mxrg):
        if ireg > 0:
            dro = data.rd[ireg] - data.rd[ireg - 1]
        else:
            dro = data.rd[0]

        for i1 in range(data.mcpreg[ireg]):
            rbo = rb
            rb += dro / data.mcpreg[ireg]
            data.rrg[ii] = rb
            data.volrrg[ii] = 3.14159 * (rb ** 2 - rbo ** 2)
            data.idmat[ii] = ireg
            totxai = 0.0

            for ig in range(data.imax):
                data.sigtrg[ii, ig] = 0.0
                data.vsigf[ii, ig] = 0.0
                data.sigabs[ii, ig] = 0.0

                for i3 in range(data.imax):
                    data.scm[ii, ig, i3] = 0.0

                for inn in range(data.mnccel[ireg]):
                    id = data.idnc1[ireg, inn]
                    idsp = data.idcl1[id]
                    data.sigtrg[ii, ig] += data.adnc[ireg, inn] * data.sigtg[id, ig]
                    data.vsigf[ii, ig] += data.adnc[ireg, inn] * data.sigf[id, ig] * data.xnu[id, ig]
                    data.sigabs[ii, ig] += data.adnc[ireg, inn] * data.siga[id, ig]

                    for i3 in range(data.imax):
                        igsp = ig + i3
                        ndelta = ig - i3
                        if igsp <= data.imax and ig <= data.mxdwni:
                            data.scm[ii, ig, i3] += data.adnc[ireg, inn] * data.stri[ig, idsp, i3]
                        if igsp <= data.imax and ig <= data.mxdwn2:
                            data.scm[ii, ig, i3] += data.adnc[ireg, inn] * data.str2[ig, idsp, i3]

                data.xai[ii, ig] = data.chi[ig, data.nchi - 1]
                totxai += data.xai[ii, ig]
            

            # print(f'init. total xai for region: {ii}, {totxai}')

            for ig in range(data.imax):
                data.xai[ii, ig] /= totxai
            
            # Display scattering, absorption, and other metrics
            # print(f'sigtot of reg: {ii}')
            # print([f"{data.sigtrg[ii, j2]:.5f}" for j2 in range(data.imax)])

            # print(f'vsigf of reg: {ii}')
            # print([f"{data.vsigf[ii, j2]:.5f}" for j2 in range(data.imax)])

            # print(f'sigabs of reg: {ii}')
            # print([f"{data.sigabs[ii, j2]:.5f}" for j2 in range(data.imax)])

            # print(f'xai for reg: {ii}')
            # print([f"{data.xai[ii, j2]:.5f}" for j2 in range(data.imax)])
            # print([f"{data.adnc[ii, j2]:.5f}" for j2 in range(data.mclnuc)])

            # print(f'total scattering matrix of region: {ii}')

            # for ig in range(data.imax):
            #     print(f'source group: {ig}')
            #     print([f"{data.scm[ii, ig, igt]:.5f}" for igt in range(data.imax)])
            
            # print(f'self consistency chek:')

            for ig in range(data.imax):
                data.qss[ii, ig] = 0.0
                for igt in range(data.imax):
                    data.qss[ii, ig] += data.scm[ii, ig, igt]
            
            # print(f'total scattering:')
            # print([f"{data.qss[ii, igt]:.5f}" for igt in range(data.imax)])

            for ig in range(data.imax):
                data.qss[ii, ig] += data.sigabs[ii, ig]
                data.sigtrg[ii, ig] = data.qss[ii, ig]
            
            # print(f'total scattering+absorbtion:')
            # print([f"{data.qss[ii, igt]:.5f}" for igt in range(data.imax)])

            ii += 1

    data.maxcrg = ii
    # print(f'ir,(atdreg(ir,j),j=1,mclnuc)')
    for ir in range(data.maxcrg):
        for inn in range(data.mclnuc):
            data.atdreg[ir, inn] = 0.0
        id1 = data.idmat[ir]
        for inn in range(data.mnccel[id1]):
            id2 = data.idnc1[id1, inn]
            data.atdreg[ir, id2] = data.adnc[id1, inn]
        # print(f'{ir}', end=',')
        # print([f'{data.atdreg[ir, j]:.5f}' for j in range(data.mclnuc)])
    
    # print(f'radius of regions in cell {data.maxcrg}')
    # print([f"{data.rrg[i]:.5f}" for i in range(data.maxcrg)])
    # print(f'volume of regions in cell {data.maxcrg}')
    # print([f"{data.volrrg[i]:.5f}" for i in range(data.maxcrg)])

    incoef(data)

    # print('enter calcp')

    data_pij = cp.zeros((data.maxcrg, data.maxcrg, data.imax), dtype=cp.float64).flatten()
    data_pesc = cp.zeros((data.maxcrg, data.imax), dtype=cp.float64).flatten()
    data_rrg = cp.asarray(data.rrg, dtype=cp.float64)
    data_absg = cp.asarray(data.absg.flatten()).astype(cp.float64)
    data_weigg = cp.asarray(data.weigg.flatten()).astype(cp.float64)
    data_sigtrg = cp.asarray(data.sigtrg.flatten()).astype(cp.float64)
    data_volrrg = cp.asarray(data.volrrg, dtype=cp.float64)

    # Define block and grid dimensions
    threads_per_block = (32, 3, 3)
    blocks_per_grid = (
        (data.imax + threads_per_block[0] - 1) // threads_per_block[0],
        (data.maxcrg + threads_per_block[1] - 1) // threads_per_block[1],
        (data.maxcrg + threads_per_block[2] - 1) // threads_per_block[2]
    )

    nbkf = 20

    # Launch the calcp kernel
    calcp_kernel(
        blocks_per_grid, threads_per_block,
        (
            data_pij, data_rrg, data_absg, data_weigg, data_sigtrg, data_volrrg,
            data.nord, data.maxcrg, data.imax, nbkf
        )
    )

    threads_per_block = (32, 3)
    blocks_per_grid = (
        (data.imax + threads_per_block[0] - 1) // threads_per_block[0],
        (data.maxcrg + threads_per_block[1] - 1) // threads_per_block[1]
    )

    # Launch the escpr kernel
    escpr_kernel(
        blocks_per_grid, threads_per_block,
        (
            data_pesc, data_rrg, data_absg, data_weigg, data_sigtrg, data_volrrg,
            data.nord, data.maxcrg, data.imax, nbkf
        )
    )

    # Copy results back to CPU
    data.pij = cp.asnumpy(data_pij).reshape((data.maxcrg, data.maxcrg, data.imax))
    data.pesc = cp.asnumpy(data_pesc).reshape((data.maxcrg, data.imax))
    # print(f'ii,(pij(ii,i3,ig),i3=1,{data.maxcrg})')

    # for ig in range(data.imax):
        # print(f'energy group: {ig}')
        # print(f'escape probability pesc(i), i=1, {data.maxcrg}')
        # print([f"{data.pesc[i3, ig]:.5f}" for i3 in range(data.maxcrg)])
        # print('collision prob. pij i:! j ->')
        # for ii in range(data.maxcrg):
            # print(f'{ii}', end=',')
            # print([f"{data.pij[ii, i3, ig]:.5f}" for i3 in range(data.maxcrg)])

    pijadj(data)
    # print("----------colpr1() end----------\n")

# Run the colpr1 function with the prepared data
start = time.time()
read1(r'D:\Tesis Ghulam Abrar\contohinput.inp', data)
timeread1 = time.time()-start
start1 = time.time()
rdlbsp(r'D:\Tesis Ghulam Abrar\libcel.dat', data)
timerdlbsp = time.time()-start1
start2 = time.time()
homog(data)
timehomog = time.time()-start2
start3 = time.time()
colpr1(data)
timecolpr1 = time.time()-start3
print('timeread1 = ', timeread1)
print('timerdlbsp = ', timerdlbsp)
print('timehomog = ', timehomog)
print('timecolpr1 = ', timecolpr1)
print('time total = ', time.time() - start)