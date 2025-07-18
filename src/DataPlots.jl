VERSION < v"0.1.0" && __precompile__()

module DataPlots

export plot_BC
export plot_proton
export plot_pbar
export plot_primary
export plot_secondary
export plot_e
export plot_pbarp
export plot_he
export plot_he34
export plot_be109
export plot_all
export dict_bump
export modulation
export +
export /
export zero_
export purety
export mod_model
export rescale

using Plots
using Interpolations
using FITSUtils
using Printf
using FITSIO
using LsqFit
using DelimitedFiles
using CubicSplines
"""
    modulation(ene::Array{T,1} where {T<:Real}, flux::Array{T,1} where {T<:Real};
               A::Int = 1, Z::Int = 1, phi::Real = 0)

Doing solar modulation for specified spectrum; ene in unit [GeV]; flux is the nucleon flux

# Arguments
* `A`,`Z`:  the A and Z of the particle
* `phi`:    modulation potential [unit: GV]
"""
function modulation(ene::Array{T,1} where {T<:Real}, flux::Array{T,1} where {T<:Real}; A::Int = 1, Z::Int = 1,phi::Real = 0)
  phi_ = phi* abs(Z) / max(A,1)
  m0 = A == 0 ? 0.511e-3 : 0.9382
  spline = CubicSpline(log.(ene), log.(flux))
  logene=log.(ene)
  eaxis = range(logene[1],stop=last(logene),length=length(logene)*10)
  faxis = spline(eaxis)
  #itpspec = interpolate((log.(ene),), log.(flux), Gridded(Linear()))
  itpspec = interpolate((eaxis,), faxis, Gridded(Linear()))
  spec = extrapolate(itpspec, Line())
  (ene, map(e-> e * (e + 2 * m0) / ( (e + phi_) * (e + phi_ + 2 * m0)) * exp(spec(log(e + phi_))), ene)) 
end
###############################################
#from 2403.20038 Wei-Cheng Long and Juan Wu's  linear-logarithm adaption of FFA
# we add a beta fector to modify the phi below 2GV
function linlog_modulation(ene::Array{T,1} where {T<:Real}, flux::Array{T,1} where {T<:Real}; A::Int = 1, Z::Int = 1,phi::Real = 0,phi2::Real = 0.03)
  m0 = A == 0 ? 0.511e-3 : 0.9382
  E_nucleon = ene.+m0
  E         = max(A,1)*E_nucleon
  pMc       =@. sqrt(E^2-m0^2*max(A,1)^2)
  rigidity  = pMc/abs(Z)
  beta      = pMc./E
  phi_n(E)=phi-phi2*log(max(A,1)*(E+m0)/abs(Z))   #log(rigidity/beta)
  phi_(E) = phi_n(E)* abs(Z) / max(A,1)

  logene = log.(ene)
  itpspec = interpolate(((log.(ene),),), log.(flux), Gridded(Linear()))
  spec = extrapolate(itpspec, Line())

  (ene, map(e-> e * (e + 2 * m0) / ( (e + phi_(e)) * (e + phi_(e) + 2 * m0)) * exp(spec(log(e + phi_(e)))), ene)) 
end
#################################################################
"""
    helmod_modulation(particle::Particle;t0::Int , t1::Int )

    given by www.helmod.org; 

# Arguments
* `t0`and`t1`:    when did this experiment start and end, YYYYMMDD
* `Z`:    only specified when modulating antiproton or electron(z=-1)
"""
function helmod_modulation(path::String;t0::Int = 20110519, t1::Int = 20160526)
  t0_br=YMDtoBR(t0)
  t1_br=YMDtoBR(t1)
  total = 0
  for t=t0_br:t1_br
  run(`python3.7 /home/dm/zhaomj/software/helmod/SolarModulation_Galprop_fast_v1.102.py -a /home/dm/zhaomj/software/helmod/HelModOnlineTime_Z-1_02 --LIS $path --SimName Proton_CR$(t)TKO -o /home/dm/zhaomj/software/helmod/output`)
  total=readdlm("/home/dm/zhaomj/software/helmod/output/ModSpectra_HelModOutput_Proton_CR$(t)TKO_Tkin.dat", ' ', Float64; comments=true, comment_char='#')[:,2].+total
  end
  ekin=readdlm("/home/dm/zhaomj/software/helmod/output/ModSpectra_HelModOutput_Proton_CR$(t0_br)TKO_Tkin.dat", ' ', Float64; comments=true, comment_char='#')[:,1]
  (ekin,total/(t1_br-t0_br+1))
end
#################################################################
"""
    cholis_modulation(particle::Particle;t0::Int , t1::Int )

Doing solar modulation for specified particle using time , rigidity and charge-dependent solar modulation from Ilias Cholis et al. arx:1511.01507; 

# Arguments
* `particle`:  a struct for one kind of particle, if multiple particles are combined ,make sure the main Isotope is added first (example: spec2['Carbon_12']+spec2['Carbon_13'],
not spec2['Carbon_13']+spec2['Carbon_12'])
* `t0`and`t1`:    when did this experiment start and end, YYYYMMDD
* `Z`:    only specified when modulating antiproton or electron(z=-1)
"""
function cholis_modulation(particle::Particle;t0::Int = 20110519, t1::Int = 20160526)
  path1="/home/dm/zhaomj/software/source/solar-modulation/example_input_files/"
  path2="ism_spectrum.txt"
  write_spec(path1*path2,particle.Ekin,particle.dNdE)
  name=(particle.Z==-1 ? (particle.A==1 ? "antiproton" : "electron" ) : 
        particle.Z==1 ? (particle.A==1 ? "proton" : (particle.A==-1 ? "deuteron" : "positron" )) :  
        particle.Z==2 ? (particle.A==3 ? "helium3" : "helium4" ) :
        particle.Z==3 ? (particle.A==6 ? "lithium6" : "lithium7" ) : 
	particle.Z==4 ? (particle.A==9 ? "beryllium9" : "beryllium10" ) : 
	particle.Z==5 ? (particle.A==10 ? "boron10" : "boron11" ) :
	particle.Z==6 ? (particle.A==12 ? "carbon12" : "carbon13" ) :
	particle.Z==7 ? (particle.A==14 ? "nitrogen14" : "nitrogen15" ) :
	particle.Z==8 ? (particle.A==16 ? "oxygen16" : (particle.A==17 ? "oxygen17" : "oxygen18" ) ) : "?")
  t0_mjd=string(YMDtoMJD(t0))
  t1_mjd=string(YMDtoMJD(t1))
  change="s/particle_name: '.*'/particle_name: '$name'/g"
  change2="s/observation_starttime: .*#/observation_starttime: $t0_mjd #/g"
  change3="s/observation_endtime: .*#/observation_endtime: $t1_mjd #/g"
  run(`sed -i $change /home/dm/zhaomj/software/source/solar-modulation/analysis_options.yaml`)
  run(`sed -i $change2 /home/dm/zhaomj/software/source/solar-modulation/analysis_options.yaml`)
  run(`sed -i $change3 /home/dm/zhaomj/software/source/solar-modulation/analysis_options.yaml`)
  run(`python3.6 /home/dm/zhaomj/software/source/solar-modulation/modulate_main.py`)
  readdlm("/home/dm/zhaomj/software/source/solar-modulation/outfile/modulatd_spectrum.txt", ' ', Float64; comments=true, comment_char='#')
end

function fast_modulation(particle::Particle)
  path1="/home/dm/zhaomj/software/source/solar-modulation/example_input_files/"
  path2="ism_spectrum.txt"
  write_spec(path1*path2,particle.Ekin,particle.dNdE)
  run(`python3.6 /home/dm/zhaomj/software/source/solar-modulation/modulate_main.py`)
  readdlm("/home/dm/zhaomj/software/source/solar-modulation/outfile/modulatd_spectrum.txt", ' ', Float64; comments=true, comment_char='#')
end

function YMDtoMJD(t::Int)
   Y=t÷1e4-1900
   M=t÷100%100
   D=t%100
   L=(M==1 || M==2) ? 1 : 0
   return MJD=14956+D+(Y-L)*365.25÷1+(M+1+L*12)*30.6001÷1
end
function YMDtoBR(t::Int)
   Y=t÷1e4-1832
   M=t÷100%100
   D=t%100
   mtd=[31,28.25,31,30,31,30,31,31,30,31,30,31]#18320226~0 bartel
   return BR=Int32(round((Y*365.25+sum(mtd[1:M])+D-57)/27))
end

function write_spec(fname::String,ene::Array{T,1} where {T<:Real}, dNdE::Array{T,1} where {T<:Real})
    open(fname, "w") do io
       writedlm(io, [ene dNdE])
    end
end
###############################################################
"""
    modulation(particle::Particle, phi::Real)

Doing solar modulation for specified spectrum inside the Particle structure

# Arguments
* `phi`:    modulation potential [unit: GV]
"""
function modulation(particle::Particle, phi::Real,phi0::Real)
  if modulation_model=="FFA"
    particle.Ekin, particle.dNdE = modulation(particle.Ekin, particle.dNdE; A=particle.A, Z=particle.Z, 
    phi= phi0!=0 && particle.Z==-1 ? phi0 : phi)
  #elseif modulation_model=="linlog"
    #particle.Ekin, particle.dNdE = linlog_modulation(particle.Ekin, particle.dNdE; A=particle.A, Z=particle.Z, phi=phi,phi2=phi0)
  end
  count_R(particle)
end

"""
    dict_modulation(spec::Dict{String,Particle}, phi::Real = 0)

Doing solar modulation for a spectra dict of many Particles produced by FITSUtils

# Arguments
* `phi`:    modulation potential [unit: GV]
* `phi0`:   modulation potential for pbar [unit: GV] or for other model param
"""
function dict_modulation(spec::Dict{String,Particle}, phi::Real = 0; phi0::Real = 0)
  phi == 0 && return spec

  mod_spec = Dict{String,Particle}()
    for k in keys(spec)
      mod_spec[k] = modulation(copy(spec[k]), phi, phi0)
    end
  return mod_spec
end

function get_data(fname::String; index::Real = 0.0, norm::Real = 1.0)
  basedir = dirname(@__FILE__)
  @assert isfile("$basedir/$fname")

  result = Dict{String, Array{Float64,2}}()
  key = ""

  open("$basedir/$fname") do file
    while !eof(file)
      line = readline(file)
      if line[1] == '#'
        key = line[2:length(line)]
        result[key] = Array{Float64,2}(undef, 0, 3)
      else
        if (key != "")
          lvec = map(x->parse(Float64, x), split(line))
          lvec[2:3] = map(v->v*lvec[1]^-index * norm, lvec[2:3])
          result[key] = vcat(result[key], lvec')
        end
      end
    end
  end

  result
end

function plot_data(data::Array{T,2} where { T <: Real },label::String)
  scatter(data[:,1], data[:,2]; yerror=(min.(data[:,3],data[:,2].*0.999),data[:,3]), linewidth=0, marker=:dot, label=label,gridalpha=0.3,gridstyle=:dash,thickness_scaling = 2,markersize = 5,markerstrokewidth=1)
end

function plot_data!(data::Array{T,2} where { T <: Real },label::String,mar::Symbol)
  scatter!(data[:,1], data[:,2]; yerror=(min.(data[:,3],data[:,2]*0.999),data[:,3]), label=label,gridalpha=0.3,gridstyle=:dash,thickness_scaling = 2,marker=mar,markersize = 5,markerstrokewidth=0.7)
end

function plot_comparison(plot_func, spectra::Array{Dict{String,Particle},1}, label::Array{String,2};
                         phi::Real = 0,phi0::Real = 0,
                         data::Array{String,1} = [],
                         datafile::String = "", index::Real = 0, norm::Real = 1,
                         xscale::Symbol = :log10, yscale::Symbol = :none,
                         ylabel::String = "")
  if phi==0 
    label.*=", LIS"
  else label.*=", phi="*string(round(phi; sigdigits=4))
  end
  if spectra[1]["Hydrogen_1"].dNdE[1]==0
   label=""
  end
  whole_ekin = true
  whole_rigidity = false
  if length(data) != 0
    pdata = get_data(datafile; index=index, norm=norm)

    if !mapreduce(k->haskey(pdata, k), &, data)
      data_keys = keys(pdata)
      throw("The specified data not found, note that the available data in file $datafile are:\n$data_keys")
    end
    if pure<1
     markers = filter((m->begin
                 m in Plots.supported_markers()
             end), Plots._shape_keys)
    markers = permutedims(markers)
   # n = length(markers)
     for i=1:length(data)
        k=data[i]
       if pure==-1
         mar_typ=markers[marker_i]
       else mar_typ=markers[i]
       end
       plot_data!(pdata[k], k,mar_typ)
     end
    end
    whole_rigidity = mapreduce(k->occursin("rigidity", k), &, data)
    whole_ekin = mapreduce(k->!occursin("rigidity", k), &, data)
  end

  if !whole_rigidity && !whole_ekin
    throw("The specified data should be all in rigidity or all in Ekin")
  end
  plot!(xscale=xscale, xlabel=whole_ekin ? "Kinetic Energy per nucleon[GeV/n]" : "Rigidity R[GV]", yscale=yscale, ylabel=whole_ekin ? ylabel : replace(replace(ylabel,"(GeV/n)"=>"GV"),"E"=>"R"))

  mod_spectra = map(spec->dict_modulation(spec,phi;phi0=phi0), spectra)
  lis=palette(:tab20)
  n=length(lis)
  for i in 1:length(mod_spectra)
    ptc = plot_func(mod_spectra[i])*norm
    ene, flux = whole_ekin ? (ptc.Ekin, ptc.dNdE) : (ptc.R, ptc.dNdR)
  #  if pure!=0
  #   plot!(ene, flux; alpha=0.1,color=color,label="",framestyle =:box)
   # else
   if phi==0 
    plot!(ene, flux; label = i<=length(label) ? label[1,i] : "",line=(:dot,1.5),framestyle =:box,size=(1600,800),legend_position=:bottomright)#,palette=[lis[n-i]])
    else 
    plot!(ene, flux; label = i<=length(label) ? label[1,i] : "",w=1,framestyle =:box,size=(1600,800),legend_position=:bottomright)#,ribbon=(flux*0.081,flux*0.092),linewidth=0)#,palette=[lis[n-i]])
    end
    
  end
  plot!()
end

function get_func(a::Particle)
  etpE = extrapolate(interpolate((log.(a.Ekin),), log.(a.dNdE), Gridded(Linear())), Line())
  etpR = extrapolate(interpolate((log.(a.R),), log.(a.dNdR), Gridded(Linear())), Line())

  (x->exp(etpE(log(x))), x->exp(etpR(log(x))))
end

function op_particle(a::Particle, b::Particle, op)
  c = copy(a)
  efunc, rfunc = get_func(b)

  c.dNdE = @. op(c.dNdE, efunc(c.Ekin))
  c.dNdR = @. op(c.dNdR, rfunc(c.R))
  c
end
function op_dict(a::Dict{String,Particle}, b::Dict{String,Particle}, op)
  c = copy(a)
  for k in collect(keys(a))
    if k=="primary_positrons"
      c[k]=op_particle(a[k], b["primary_electrons"], op)
    else 
      c[k]=op_particle(a[k], b[k], op)
    end
  end
  c
end
Base.:+(a::Particle, b::Particle) = op_particle(a, b, +)
Base.:-(a::Particle, b::Particle) = op_particle(a, b, -)
Base.:*(a::Particle, b::Particle) = op_particle(a, b, *)
Base.:/(a::Particle, b::Particle) = op_particle(a, b, /)
Base.:+(a::Dict{String,Particle}, b::Dict{String,Particle}) = op_dict(a, b, +)
Base.:-(a::Dict{String,Particle}, b::Dict{String,Particle}) = op_dict(a, b, -)

function Base.:*(a::Particle, n::Real)
  a.dNdE .*= n
  a.dNdR .*= n
  a
end

function rescale(a::Particle, index::Real)
  b = deepcopy(a)
  b.dNdE = @. b.dNdE * b.Ekin^index
  b.dNdR = @. b.dNdR * b.R^index
  b
end

function rescale(spec::Dict{String,Particle}, norm::Real; z::Int=0,index::Real=0)
  nspec = deepcopy(spec)
  for a in collect(keys(spec))
   if z==0
  nspec[a].dNdE = @. nspec[a].dNdE * norm * nspec[a].Ekin^index
  nspec[a].dNdR = @. nspec[a].dNdR * norm * nspec[a].R^index
   elseif nspec[a].Z==z
    nspec[a].dNdE = @. nspec[a].dNdE * norm * nspec[a].Ekin^index
    nspec[a].dNdR = @. nspec[a].dNdR * norm * nspec[a].R^index
   end
  end
  nspec
end
function rescale_sec(spec::Dict{String,Particle}, norm::Real,index::Real)
   nspec = deepcopy(spec)
   nspec =rescale(nspec,norm,z=3,index=-index)
   nspec =rescale(nspec,norm,z=4,index=-index)
   nspec =rescale(nspec,norm,z=5,index=-index)
   nspec =rescale(nspec,norm,z=9,index=-index)
   nspec
end

function zero_(spec::Dict{String,Particle})
  rescale(spec,0)
end

"""
    plot_BC(spectra::Array{Dict{String,Particle},1}, label::Array{String,2};
            phi::Real = 0, data::Array{String,1})  

    Ploting the B/C ratio of given spectra in comparison with the data

# Arguments
* `phi`:     modulation potential [unit: GV]
* `data`:    The dataset to plot
"""
function plot_BC(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0)); phi::Real = 0,phi0::Real = 0, data::Array{String,1}=["AMS02(2011/05-2016/05)"])
  plot_comparison(spec-> (spec["Boron_10"] + spec["Boron_11"]) / (spec["Carbon_12"] + spec["Carbon_13"]),
                  spectra, label; phi=phi,phi0=phi0, data=data, datafile="bcratio.dat", ylabel="B/C ratio")
end

function plot_BeB(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0)); phi::Real = 0,phi0::Real = 0, data::Array{String,1}=["AMS02rigidity(2011/05-2016/05)"])
 plot_comparison(spec-> (spec["Beryllium_7"] + spec["Beryllium_9"] + spec["Beryllium_10"]) / (spec["Boron_10"] + spec["Boron_11"]),
#plot_comparison(spec-> (spec["Beryllium_10"]) / (spec["Boron_10"] + spec["Boron_11"]),
                  spectra, label; phi=phi,phi0=phi0, data=data, datafile="bebratio.dat", ylabel="Be/B ratio")
end

function plot_doubleratio(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0));a::String="Fe",b::String="O", phi::Real = 0,phi0::Real = 0, data::Array{String,1}=["ams"],k::Real = 1)
  ylabel="F/Si:B/O"
  data=["AMS02(F/Si:B/O)rigidity"]
  _func(x)=fun_ratio(x,a="F",b="Si")/fun_ratio(x,a="B",b="O")
  plot_comparison(_func,spectra, label; phi=phi,phi0=phi0, data=data, datafile="ratio.dat", ylabel=ylabel*" ratio")
end

function plot_ratio(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0));a::String="Fe",b::String="O", phi::Real = 0,phi0::Real = 0, data::Array{String,1}=["ams"],k::Real = 1)
 ylabel=a*"/"*b
 data=(data==["rigidity"] ? ["AMS2021(Fe/O)rigidity(2011/05/19-2019/10/30)"] : 
       data==["ekin"] ? ["HEAO3-C2(Fe/O)(1979/10-1980/06)"] : data)
 if data==["ams"]
   data=(ylabel=="Na/Si" ? ["AMS2021(Na/Si)rigidity(2011/05/19-2019/10/30)"] : 
   ylabel=="Al/Si" ? ["AMS2021(Al/Si)rigidity(2011/05/19-2019/10/30)"] : 
   ylabel=="Na/F" ? ["AMS2021(Na/F)rigidity(2011/05/19-2019/10/30)"] : 
   ylabel=="Al/F" ? ["AMS2021(Al/F)rigidity(2011/05/19-2019/10/30)"] : 
   ylabel=="Fe/O" ? ["AMS2021(Fe/O)rigidity(2011/05/19-2019/10/30)"] : 
   ylabel=="Fe/He" ? ["AMS2021(Fe/He)rigidity(2011/05/19-2019/10/30)"] : 
   ylabel=="Fe/Si" ? ["AMS2021(Fe/Si)rigidity(2011/05/19-2019/10/30)"] : 
   ylabel=="Ne/Mg" ? ["AMS2020(Ne/Mg)rigidity(2011/05-2018/05)"] : 
   ylabel=="B/O" ? ["AMS2018(B/O)rigidity(2011/05-2016/05)SM3300"] : 
   ylabel=="Be/C" ? ["AMS2018(Be/C)rigidity(2011/05-2016/05)"] : 
   ylabel=="Be/O" ? ["AMS2018(Be/O)rigidity(2011/05-2016/05)"] : 
   ylabel=="C/O" ? ["AMS2020(C/O)rigidity(2011/05-2018/05)"] : 
   ylabel=="He/O" ? ["AMS2020(He/O)rigidity(2011/05-2018/05)"] : 
   ylabel=="Li/B" ? ["AMS2018(Li/B)rigidity(2011/05-2016/05)"] : 
   ylabel=="Li/C" ? ["AMS2020(Li/C)rigidity(2011/05-2018/05)"] : 
   ylabel=="Li/O" ? ["AMS2020(Li/O)rigidity(2011/05-2018/05)"] : 
   ylabel=="Mg/O" ? ["AMS2020(Mg/O)rigidity(2011/05-2018/05)"] : 
   ylabel=="N/B" ? ["AMS2020(N/B)rigidity(2011/05-2018/05)"] : 
   ylabel=="N/O" ? ["AMS2020(N/O)rigidity(2011/05-2018/05)"] : 
   ylabel=="Ne/O" ? ["AMS2020(Ne/O)rigidity(2011/05-2018/05)"] : 
   ylabel=="Si/Mg" ? ["AMS2020(Si/Mg)rigidity(2011/05-2018/05)"] : 
   ylabel=="Si/O" ? ["AMS2020(Si/O)rigidity(2011/05-2018/05)"] : 
   ylabel=="F/Si" ? ["AMS2021(F/Si)rigidity(2011/05/19-2019/10/30)"] : 
   ylabel=="F/B" ? ["AMS2021(F/B)rigidity(2011/05/19-2019/10/30)"] : 
   ylabel=="p/He" ? ["AMS2020(p/He)rigidity(2011/05-2018/05)"] : 
    ylabel=="D/p" ? ["AMS2024(D/p)rigidity(2011/05-2021/04)"] : 
     ylabel=="D/4He" ? ["AMS2024(D/4He)rigidity(2011/05-2021/04)"] : 
         ylabel=="B/C" ? ["AMS02(B/C)rigidity(2011/05-2016/05)SM3300"] : 
         ylabel=="Be/B" ? ["AMS02(Be/B)rigidity(2011/05-2016/05)"] : 
         ylabel=="7Li/6Li" ? ["AMS2023(7Li/6Li)pre"] : 
         ylabel=="e+/eall" ? ["AMS2019fraction(2011/05/19-2017/11/12)"] : 
        ylabel=="SubFe/Fe" ? ["AMS2023(SubFe/Fe)pre"] : data)
 end
  _func(x)=fun_ratio(x,a=a,b=b)*k
  plot_comparison(_func,spectra, label; phi=phi,phi0=phi0, data=data, datafile="ratio.dat", ylabel=ylabel*" ratio")
end
"""
    plot_proton(spectra::Array{Dict{String,Particle},1}, label::Array{String,2};
            phi::Real = 0, data::Array{String,1})  

    Ploting the proton flux of given spectra in comparison with the data

# Arguments
* `phi`:     modulation potential [unit: GV]
* `data`:    The dataset to plot
"""
function plot_proton(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0)); phi::Real = 0,phi0::Real = 0, data::Array{String,1}=["AMS02rigidity(2011/05-2018/05)"],k::Real = 1)
 k != 1 && (label.*=",k="*string(k))
 #["AMS02(2011/05-2018/05)","DAMPE2019(2016/01/01-2018/06/30)","CALET(2015/10-2021/12)","CREAM-I+III(2004+2007)","NUCLEON-KLEM(2015/07-2017/06)","TUNKA-133ArrayQGSJet01(2009/10-2012/04)","IceCube-IceTop(2019)","KG_QGSjet-II-02(2013)","KG_SIBYLL-23(2017)","KAS_SIBYLL-21(2011)","KAS_EPOS-199(2011)","KAS_QGSjet-II-02(2011)"]
  plot_comparison(spec -> rescale(spec["Hydrogen_1"]+spec["secondary_protons"]+(haskey(spec,"pp_deuterons") ? spec["pp_deuterons"]+spec["Hydrogen_2"] : spec["Hydrogen_2"]), 2.7) * 1e4,
                  spectra, label; phi=phi,phi0=phi0, data=data, datafile="proton.dat", index=0,norm=k,yscale=:log10, ylabel="\$\\rm E^{2.7}dN/dE [m^{-2}sr^{-1}s^{-1}(GeV/n)^{1.7}]\$")
end

function plot_primary(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0)); phi::Real = 0,phi0::Real = 0, data::Array{String,1}=["AMS2017heliumrigidity(2011/05/19-2016/05/26)"],k::Real = 1)
#["AMS02helium(2011/05-2018/05)","DAMPE2021helium(2016/01/01-2020/06/30)","CALEThelium(2015/10-2022/04)","CREAM2017helium(2007/12-2008/01)","NUCLEON-KLEMhelium(2015/07-2017/06)","TUNKA-133-QGSJet01helium(2009/10-2012/04)","ICE-Cubehelium(2019)","KG_QGSjet-II-02helium(2013)","KAS_QGSjet01helium(2011)","KAS_QGSjet01helium(2005)","KAS_SIBYLL21helium(2005)"]
  data=(data==["he"] ? ["AMS02heliumrigidity(2011/05-2018/05)"] : 
        data==["c"]  ? ["AMS2023carbonrigidity(2011/05-2021/05)"] :
        data==["o"]  ? ["AMS2023oxygenrigidity(2011/05-2021/05)"] :
        data==["n"]  ? ["AMS2018nitrogenrigidity(2011/05/19-2016/05/26)"] : 
		data==["ne"]  ? ["AMS2023neonrigidity(2011/05-2021/05)"] : 
		data==["mg"]  ? ["AMS2023magnesiumrigidity(2011/05-2021/05)"] : 
		data==["si"]  ? ["AMS2023siliconrigidity(2011/05-2021/05)"] :
		data==["na"]  ? ["AMS2021sodiumrigidity(2011/05/19-2019/10/30)"] :
		data==["al"]  ? ["AMS2021aluminiumrigidity(2011/05/19-2019/10/30)"] :
		data==["s"]  ? ["AMS2023sulfurrigidity(2011/05-2021/05)"] :
		data==["ar"]  ? ["HEAO3-C2argon(1979/10-1980/06)"] :
		data==["ca"]  ? ["HEAO3-C2calcium(1979/10-1980/06)"] :
		data==["fe"]  ? ["AMS2021ironrigidity(2011/05/19-2019/10/30)"] : data)
  _func=(occursin("helium", data[1])  ? spec -> rescale(spec["Helium_3"] + spec["Helium_4"], 2.7) * 1e4 : 
         occursin("carbon", data[1])  ? spec -> rescale(spec["Carbon_12"] + spec["Carbon_13"], 2.7) * 1e4 :
         occursin("oxygen", data[1])  ? spec ->rescale(spec["Oxygen_16"] + spec["Oxygen_17"] + spec["Oxygen_18"], 2.7) * 1e4 :
         occursin("nitrogen", data[1]) ? spec ->rescale(spec["Nitrogen_14"] + spec["Nitrogen_15"], 2.7) * 1e4 :
		 occursin("neon", data[1]) ? spec ->rescale(spec["Neon_20"] + spec["Neon_21"] + spec["Neon_22"], 2.7) * 1e4 :
		 occursin("magnesium", data[1]) ? spec ->rescale(spec["Magnesium_24"] + spec["Magnesium_25"] + spec["Magnesium_26"], 2.7) * 1e4 :
		 occursin("silicon", data[1]) ? spec ->rescale(spec["Silicon_28"] + spec["Silicon_29"] + spec["Silicon_30"], 2.7) * 1e4 : 
		 occursin("sodium", data[1]) ? spec ->rescale(spec["Sodium_23"] , 2.7) * 1e4 : 
		 occursin("aluminium", data[1]) ? spec ->rescale(spec["Aluminium_26"] + spec["Aluminium_27"], 2.7) * 1e4 : 
		 occursin("sulfur", data[1]) ? spec ->rescale(spec["Sulphur_32"] + spec["Sulphur_33"]+spec["Sulphur_34"], 2.7) * 1e4 : 
		 occursin("argon", data[1]) ? spec ->rescale(spec["Argon_36"] + spec["Argon_37"]+spec["Argon_38"]+spec["Argon_40"], 2.7) * 1e4 : 
		 occursin("calcium", data[1]) ? spec ->rescale(spec["Calcium_40"] + spec["Calcium_41"]+spec["Calcium_42"]+spec["Calcium_43"] + spec["Calcium_44"]+spec["Calcium_46"]+spec["Calcium_48"], 2.7) * 1e4 : 		 
		 occursin("iron", data[1]) ? spec ->rescale(spec["Iron_54"]+spec["Iron_55"] + spec["Iron_56"] + spec["Iron_57"]+spec["Iron_58"]+spec["Iron_60"], 2.7) * 1e4 :
		 occursin("nickel", data[1]) ? spec ->rescale(spec["Nickel_56"] + spec["Nickel_58"] + spec["Nickel_59"]+ spec["Nickel_60"]+ spec["Nickel_61"]+ spec["Nickel_62"]+ spec["Nickel_64"], 2.7) * 1e4 : spec -> rescale(spec["Helium_3"] + spec["Helium_4"], 2.7) * 1e4)
		 k != 1 && (label.*=",k="*string(k))
  plot_comparison(_func,spectra, label; phi=phi,phi0=phi0, data=data, datafile="primary.dat", index=-2.7,norm=k,yscale=:log10, ylabel="\$\\rm E^{2.7}dN/dE [m^{-2}sr^{-1}s^{-1}(GeV/n)^{1.7}]\$")
end

function plot_secondary(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0)); phi::Real = 0,phi0::Real = 0, data::Array{String,1}=["AMS2017lithiumrigidity(2011/05/19-2016/05/26)"],k::Real = 1)
  data=(data==["li"] ? ["AMS2017lithiumrigidity(2011/05/19-2016/05/26)"] : 
        data==["li6"] ? ["AMS2022li6pre"] : 
        data==["li7"] ? ["AMS2022li7pre"] : 
        data==["be"] ? ["AMS2017berylliumrigidity(2011/05/19-2016/05/26)"] :
        data==["be9"] ? ["AMS2023be9pre"] :
        data==["be10"] ? ["AMS2023be10pre"] :
        data==["be7"] ? ["AMS2023be7pre"] :
        data==["b"]  ? ["AMS2017boronrigidity(2011/05/19-2016/05/26)"] : 
        data==["f"]  ? ["AMS2021fluorinerigidity(2011/05-2019/10)"] : 
        data==["d"]  ? ["PAMELA-TOFdeuteron(2006/07-2007/12)"] :
        data==["p"]  ? ["AMS2023phosphorusrigiditypre"] :
        data==["sc"]  ? ["HEAO3-C2scandium(1979/10-1980/06)"] :
        data==["ti"]  ? ["HEAO3-C2titanium(1979/10-1980/06)"] :
        data==["v"]  ? ["HEAO3-C2vanadium(1979/10-1980/06)"] : data)
  _func=(occursin("lithium", data[1])  ? spec ->rescale(spec["Lithium_6"] + spec["Lithium_7"], 2.7) * 1e4 : 
         occursin("li6", data[1])  ? spec ->rescale(spec["Lithium_6"] , 2.7) * 1e4 : 
         occursin("li7", data[1])  ? spec ->rescale(spec["Lithium_7"] , 2.7) * 1e4 : 
         occursin("beryllium", data[1]) ? spec -> rescale(spec["Beryllium_7"] + spec["Beryllium_9"] + spec["Beryllium_10"], 2.7) * 1e4 :
         occursin("be9", data[1]) ? spec -> rescale(spec["Beryllium_9"], 2.7) * 1e4 :
         occursin("be10", data[1]) ? spec -> rescale(spec["Beryllium_10"], 2.7) * 1e4 :
         occursin("be7", data[1]) ? spec -> rescale(spec["Beryllium_7"], 2.7) * 1e4 :
         occursin("boron", data[1])    ? spec -> rescale(spec["Boron_10"] + spec["Boron_11"], 2.7) * 1e4 : 
         occursin("fluorine", data[1])    ? spec -> rescale(spec["Fluorine_19"], 2.7) * 1e4 : 
         occursin("deuteron", data[1])    ? spec -> rescale(haskey(spec,"pp_deuterons") ? spec["pp_deuterons"]+spec["Hydrogen_2"] : spec["Hydrogen_2"], 2.7) * 1e4 : 
         occursin("phosphorus", data[1])    ? spec -> rescale(spec["Phosphorus_31"], 2.7) * 1e4 :
         occursin("scandium", data[1])    ? spec -> rescale(spec["Scandium_45"], 2.7) * 1e4 :
         occursin("titanium", data[1])    ? spec -> rescale(spec["Titanium_44"]+spec["Titanium_46"]+spec["Titanium_47"]+spec["Titanium_48"]+spec["Titanium_49"]+spec["Titanium_50"], 2.7) * 1e4 :
         occursin("vanadium", data[1])    ? spec -> rescale(spec["Vanadium_49"]+spec["Vanadium_50"]+spec["Vanadium_51"], 2.7) * 1e4 : spec ->rescale(spec["Lithium_6"] + spec["Lithium_7"], 2.7) * 1e4 )
          k != 1 && (label.*=",k="*string(k))
  plot_comparison(_func,spectra, label; phi=phi,phi0=phi0, data=data, datafile="secondary.dat", index=-2.7,norm=k,yscale=:log10, ylabel="\$\\rm E^{2.7}dN/dE [m^{-2}sr^{-1}s^{-1}(GeV/n)^{1.7}]\$")
end
"""
(under test)
    plot_all(spectra::Array{Dict{String,Particle},1}, label::Array{String,2};
            phi::Real = 0, data::Array{String,1})  

    Ploting all particles flux in comparison with the data,in Etot(=Ekin/n*A+m0*A)
      SIBYLL model seems to be favoured
# Arguments
* `data`:    The dataset to plot
"""
function plot_all(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0)); phi::Real = 0,phi0::Real = 0, data::Array{String,1}=[""],species::Array{T,1} where {T<:Real}=[0])
#["NUCLEON-KLEM(2015/07-2017/06)","HAWC(2021)","Ice-Cube(2019)","Ice-Top_SIB(2020)","Ice-Top_QGS(2020)","KAS_QGSjet01(2011)","KG_SIBYLL-23(2017)","KAS+KG_QGSjet-II-04(2015)","PierreAuger-SD750(2021)","TAHybrid(2008/01-2015/05)","TUNKA-133(2020)","TibetQGS+HD(2008)","GAMMA(2014)"]
#["AMS2017(p+He)(2011/05-2018/05)","DAMPE2023(p+He)","CREAM(p+He)(2004-2005)","ARGO-YBJ+WFCT2015(p+He)","HAWC2022(p+He)","EAS-TOP+MACRO2004(p+He)","Maket-ANI(p+He)2007","KASCADE2005(p+He)","KAS+KG_QGSjet-II-04(p+He)2015"]
   mod_spectra = map(spec->dict_modulation(spec,phi), spectra)
   phe= occursin("p+He", data[1])  ? 1 : 0
   lnA= occursin("lnA", data[1])  ? 1 : 0
   m0 = 0.9382   
   pdata = get_data("allparticles.dat"; index=lnA==0 ? -2.6 : 0, norm=1)
     if phi==0 
    label=["LIS"]
  else label.*=",phi="*string(phi)
  end
  if spectra[1]["Hydrogen_1"].dNdE[1]==0
   label=""
  end
    if pure<1
     markers = filter((m->begin
                 m in Plots.supported_markers()
             end), Plots._shape_keys)
    markers = permutedims(markers)
    n = length(markers)
      for i=1:length(data)
     k=data[i]
     if pure==-1
     mar_typ=markers[marker_i]
     else mar_typ=markers[i]
     end
       plot_data!(pdata[k], k,mar_typ)
     end
    end
   
    for i in 1:length(mod_spectra)
   tot_spec = Dict{String,Array{T,1} where {T<:Real}}()
   mspec=mod_spectra[i]
    eaxis=mspec["Hydrogen_1"].Ekin
    eaxis=eaxis[eaxis .> 1e1]
  if phe==1
     for k in keys(mspec)
      if mspec[k].A>0&&(mspec[k].Z==1||mspec[k].Z==2)
       fun =  inter_fun((mspec[k].Ekin.+m0)*mspec[k].A,mspec[k].dNdE*1e4/mspec[k].A)
       tot_spec[k]=fun.(eaxis)
       end
     end
  else
   for k in keys(mspec)
    if mspec[k].A>0
     fun =  inter_fun((mspec[k].Ekin.+m0)*mspec[k].A,mspec[k].dNdE*1e4/mspec[k].A)
     tot_spec[k]=fun.(eaxis)
    end
   end 
  end
    specall=[sum([tot_spec[k][n] for k in keys(tot_spec)]) for n=1:length(eaxis)]
    #species=[1.01,2.04,6.12,8.16,10.22,12.24,14.28,26.56]
    if species!=[0]
    for k in keys(tot_spec)
      if (mspec[k].Z+mspec[k].A*0.01) in species
        plot!(eaxis, tot_spec[k].*eaxis.^2.6; label = k,w=1,framestyle =:box,line=(:dot,1.5))
      end
    end
    end
  if lnA==0
   if phi==0 
    plot!(eaxis, specall.*eaxis.^2.6; label = i<=length(label) ? label[1,i] : "",line=(:dot,1.5),framestyle =:box)#,palette=[lis[n-i]])
    else 
    plot!(eaxis, specall.*eaxis.^2.6; label = i<=length(label) ? label[1,i] : "",w=1,framestyle =:box)#,palette=[lis[n-i]])
    end  
  else
    lnaall=[sum([tot_spec[k][n]*log(mspec[k].A) for k in keys(tot_spec)]) for n=1:length(eaxis)]
    plot!(eaxis, lnaall./specall; label = i<=length(label) ? label[1,i] : "",w=1,framestyle =:box)
  end
  end 
  plot!(xscale=:log10,yscale=lnA==0 ? (:log10) : (:identity), w=1,framestyle =:box,xlabel="Total Energy [GeV]" ,ylabel=lnA==0 ? "\$\\rm E^{2.6}dN/dE [m^{-2}sr^{-1}s^{-1}(GeV)^{1.6}]\$" : "\$\\rm <ln(A)>\$")
end

function plot_e(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0)); phi::Real = 0,phi0::Real = 0, data::Array{String,1}=["AMS2019electron(2011/05/19-2017/11/12)"],k::Real = 1)
  data=(data==["e-"] ? ["AMS2019electron(2011/05/19-2017/11/12)"] : 
        data==["e+"]  ? ["AMS2019positron(2011/05/19-2017/11/12)"] :
        data==["eall"]  ? ["AMS02combined(2011/05-2018/05)"] :
        data==["fr"]  ? ["AMS2019fraction(2011/05/19-2017/11/12)"] :
        data==["pe"]  ? ["AMS02primaryele(computed)"] : data)
  _func=(occursin("electron", data[1])  ? spec -> rescale(spec["primary_electrons"]+ spec["secondary_electrons"], 3.0) * k*1e4 : 
         occursin("positron", data[1])  ? spec -> rescale(spec["secondary_positrons"] + spec["primary_positrons"], 3.0) * k*1e4 :
         occursin("computed", data[1])  ? spec -> rescale( deepcopy(spec["primary_electrons"]), 3.0) *k* 1e4 :
      #   occursin("computed", data[1])  ? spec -> rescale(spec["primary_electrons"] + spec["secondary_electrons"]-spec["secondary_positrons"], 3.0) * 1e4 :
         occursin("combined", data[1])  ? spec -> rescale(spec["primary_electrons"] + spec["secondary_electrons"]+spec["secondary_positrons"] + spec["primary_positrons"], 3.0) *k* 1e4 :
         occursin("fraction", data[1]) ? (spec->(spec["secondary_positrons"] + spec["primary_positrons"])/(spec["primary_electrons"] + spec["secondary_electrons"]+spec["secondary_positrons"] + spec["primary_positrons"])) : spec -> rescale(spec["primary_electrons"] + spec["secondary_electrons"], 3.0) *k* 1e4)
  index= occursin("fraction", data[1]) ? 0 : -3
  if occursin("fraction", data[1])
  yscale=:identity
  else yscale=:log10
  end
  plot_comparison(_func,spectra, label; phi=phi,phi0=phi0, data=data, datafile="e+e-.dat", index=index,yscale=yscale, ylabel=occursin("fraction", data[1]) ? "\$\\rm e^{+}/(e^{+}+e^{-})\\ ratio\$" : "\$\\rm E^{3}dN/dE [m^{-2}sr^{-1}s^{-1}(GeV/n)^{2}]\$")
end

function plot_he(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0)); phi::Real = 0,phi0::Real = 0, data::Array{String,1}=["AMS2019he4(2011/05-2017/11)"])
  data=(data==["he4"] ? ["AMS2019he4(2011/05-2017/11)"] : 
        data==["he3"]  ? ["AMS2019he3(2011/05-2017/11)"] :
        data==["he3r"]  ? ["AMS2019he3rigidity(2011/05-2017/11)"] :
        data==["he4r"]  ? ["AMS2019he4rigidity(2011/05-2017/11)"] : data)
_func=(occursin("he4", data[1])  ? spec -> rescale(spec["Helium_4"] , 2.7) * 1e4 : 
       occursin("he3", data[1])  ? spec -> rescale(spec["Helium_3"] , 2.7) * 1e4 : spec -> rescale(spec["Helium_4"] , 2.7) * 1e4)
  plot_comparison(_func,spectra, label; phi=phi,phi0=phi0, data=data, datafile="heratio.dat", index=-2.7,yscale=:log10, ylabel="\$\\rm E^{2.7}dN/dE [m^{-2}sr^{-1}s^{-1}(GeV/n)^{1.7}]\$")
end

"""
    plot_pbar(spectra::Array{Dict{String,Particle},1}, label::Array{String,2};
            phi::Real = 0, data::Array{String,1})  

    Ploting the antiproton flux of given spectra in comparison with the data

# Arguments
* `phi`:     modulation potential [unit: GV]
* `data`:    The dataset to plot

# phi0: AMS02rigidity(2011/05-2015/05) may be 0.42 ,BESS-PolarII(2007/12-2008/01) may be 0.38,PAMELA(2006/07-2008/12) may be 0.41,PAMELA(2006/07-2009/12) may be 0.39,BESSI(2004/12) may be 0.39 
"""
function plot_pbar(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0)); phi::Real = 0,phi0::Real = 0, data::Array{String,1}=["AMS02rigidity(2011/05-2018/05)"],k::Real = 1)           
  
  plot_comparison(spec -> rescale(spec["secondary_antiprotons"] + spec["tertiary_antiprotons"], 2.7) * 1e4*k,
                  spectra, label; phi=phi,phi0=phi0==0 ? phi : phi0, data=data, datafile="pbar.dat",index=-2.7, ylabel="\$\\rm E^{2.7}dN/dE [m^{-2}sr^{-1}s^{-1}(GeV/n)^{1.7}]\$")
 #plot_comparison(spec -> rescale(spec["DM_antiprotons"], 2.0) * 1e-3,
 #                spectra, label; phi=phi, data=data, datafile="pbar.dat",index=-2, yscale=:log, ylabel="\$E^{2}dN/dE [GeV^{2}(m^{2}*sr*s*GeV)^{-1}]\$")
end


function plot_pbarp(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0)); phi::Real = 0,phi0::Real = 0, data::Array{String,1}=["AMS02rigidity(2011/05-2018/05)"],k::Real = 1) 
  plot_comparison(spec -> (spec["secondary_antiprotons"] + spec["tertiary_antiprotons"]) / (spec["Hydrogen_1"] +spec["secondary_protons"])*k, 
                  spectra, label; phi=phi,phi0=phi0, data=data, datafile="pbarp.dat", ylabel="\$\\rm \\bar p/p\\ ratio\$")
end

function plot_be109(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0)); phi::Real = 0,phi0::Real = 0, data::Array{String,1}=["ACE-CRIS(1997/08/27-1999/04/09)","ISOMAX(1998/08/04-08/05)"])
  _func=spec -> (spec["Beryllium_10"] / spec["Beryllium_9"]) 
  plot_comparison(_func,spectra, label; phi=phi,phi0=phi0, data=data, datafile="beratio.dat", ylabel="\$\\rm ^{10}Be/^{9}Be\\ ratio\$")
end

function plot_he34(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0)); phi::Real = 0,phi0::Real = 0, data::Array{String,1}=["AMS2019rigidity(2011/05-2017/11)"])
  _func=spec -> (spec["Helium_3"] / spec["Helium_4"]) 
  plot_comparison(_func,spectra, label; phi=phi,phi0=phi0, data=data, datafile="heratio.dat", ylabel="\$\\rm ^{3}He/^{4}He\\ ratio\$")
end

function fun_ratio(spec::Dict{String,Particle};a::String="Fe",b::String="O")
   return fun_particle(spec,a)/fun_particle(spec,b)
end
function fun_particle(spec::Dict{String,Particle},a::String)
   list= (a=="p" ? ["Hydrogen_1","Hydrogen_2","secondary_protons"] : 
          a=="e+" ? ["secondary_positrons","primary_positrons"] : 
          a=="eall" ? ["primary_electrons","secondary_electrons","secondary_positrons","primary_positrons"] : 
          a=="D" ? ["Hydrogen_2"] : 
          a=="He" ? ["Helium_3" , "Helium_4"] : 
          a=="4He" ? ["Helium_4"] : 
          a=="Li" ? ["Lithium_6" , "Lithium_7"] : 
          a=="Be" ? ["Beryllium_7" , "Beryllium_9","Beryllium_10"] : 
          a=="B" ? ["Boron_10" , "Boron_11"] : 
          a=="C" ? ["Carbon_12" , "Carbon_13"] : 
          a=="N" ? ["Nitrogen_14" , "Nitrogen_15"] : 
          a=="O" ? ["Oxygen_16" , "Oxygen_17","Oxygen_18"] : 
          a=="F" ? ["Fluorine_19"] : 
          a=="Ne" ? ["Neon_20" , "Neon_21","Neon_22"] : 
          a=="Mg" ? ["Magnesium_24" , "Magnesium_25","Magnesium_26"] : 
          a=="Si" ? ["Silicon_28" , "Silicon_29","Silicon_30"] : 
          a=="Na" ? ["Sodium_23"] : 
          a=="Al" ? ["Aluminium_26" , "Aluminium_27"] :  
          a=="S" ? ["Sulphur_32" ,"Sulphur_33","Sulphur_34"] :
          a=="Fe" ? ["Iron_54" ,"Iron_55", "Iron_56","Iron_57","Iron_58","Iron_60"] : 
          a=="Ni" ? ["Nickel_56","Nickel_58","Nickel_59","Nickel_60","Nickel_61","Nickel_62","Nickel_64"] : 
          a=="SubFe" ? ["Scandium_45","Titanium_44","Titanium_46","Titanium_47","Titanium_48","Titanium_49","Titanium_50","Vanadium_49","Vanadium_50","Vanadium_51"] : 
          a=="7Li" ? ["Lithium_7"] :  
          a=="6Li" ? ["Lithium_6"] :  
          a=="7Be" ? ["Beryllium_7"] :  
          a=="9Be" ? ["Beryllium_9"] :  
          a=="10Be" ? ["Beryllium_10"] :  
          a=="22Ne" ? ["Neon_22"] :  
          a=="21Ne" ? ["Neon_21"] :  
          a=="20Ne" ? ["Neon_20"] : ["Hydrogen_1","Hydrogen_2","secondary_protons"] )
   return sum([spec[list[i]] for i=1:length(list)])
end
#############################################################
function plot_index(a::Particle,start::Int ,length::Int )
  logene = log10.(a.R)
  logflux = log10.(a.dNdR)
  gamma=fitting([logene logflux])[1]
  plot!(a.R[start:start+length],gamma[start:start+length], xscale=:log10)
end

function plot_list(spec::Dict{String,Particle};start::Int=25 ,length::Int=20)
   plot_index(modulation(spec["Hydrogen_1"]+spec["Hydrogen_2"]+spec["secondary_protons"],0.7),start+3,length)
   plot_index(modulation(spec["Helium_3"] + spec["Helium_4"],0.8),start,length)
   plot_index(modulation(spec["Carbon_12"] + spec["Carbon_13"],0.8),start,length)
   plot_index(modulation(spec["Oxygen_16"] + spec["Oxygen_17"] + spec["Oxygen_18"],0.8),start,length)
   plot_index(modulation(spec["Beryllium_7"] + spec["Beryllium_9"] + spec["Beryllium_10"],0.8),start,length)
   plot_index(modulation(spec["Boron_10"] + spec["Boron_11"],0.8),start,length)
end

function plot_exlist()
   plot_exindex("AMS2017heliumrigidity(2011/05/19-2016/05/26)","primary.dat",20,40,"he")
   plot_exindex("AMS2017carbonrigidity(2011/05/19-2016/05/26)","primary.dat",20,40,"c")
   plot_exindex("AMS2017oxygenrigidity(2011/05/19-2016/05/26)","primary.dat",20,39,"o")
   plot_exindex("AMS2017berylliumrigidity(2011/05/19-2016/05/26)","secondary.dat",20,39,"be")
   plot_exindex("AMS2017boronrigidity(2011/05/19-2016/05/26)","secondary.dat",20,39,"b")
   plot_exindex("AMS02rigidity(2011/05-2018/05)","proton.dat",25,39,"proton",index=2.7)
   plot_exindex("AMS02rigidity(2011/05-2018/05)","pbar.dat",25,25,"pbar")
end

function plot_exindex(k::String,datafile::String ,start::Int ,length::Int,label::String; index::Real = 0, norm::Real = 1)
      pdata = get_data(datafile; index=index, norm=norm)
      data=pdata[k]
  logene = log10.(data[:,1])
  logflux = log10.(data[:,2])
  gamma,err=fitting([logene logflux])
  plot!(data[:,1][start:start+length],gamma[start:start+length];ribbon=err[start:start+length],linewidth=0, marker=:dot, xscale=:log10,label=label)
end

#############################################################
# used in 2105,04630 for local bumps
function bump(rigidity::Array{T,1} where {T<:Real}, flux::Array{T,1} where {T<:Real},R0::Real, RL::Real,q::Real)
  gamma=-fitting([log10.(rigidity) log10.(flux)])[1]

  fun =  inter_fun(rigidity[4:length(rigidity)-3],max.(gamma,0.01))
  gamma=fun.(rigidity)
  (rigidity,abs.(flux.*((gamma.+2)./(q.-gamma).*exp.(-(R0./rigidity).^0.5-(rigidity/RL).^0.5).+1))) 
end

function bump(particle::Particle, R0::Real, RL::Real,q::Real)
  particle.R, particle.dNdR = bump(particle.R, particle.dNdR, R0,RL,q)
  count_Ekin(particle)
end


function dict_bump(spec::Dict{String,Particle}; R0::Real = 5878, RL::Real = 2.24e5,q::Real = 4.2)
  q == 0 && return spec

  bump_spec = Dict{String,Particle}()
  for k in keys(spec)
    bump_spec[k] = bump(copy(spec[k]), R0,RL,q)
  end
  return bump_spec
end
######################################
function fitting(xydata::Array{T,2} where { T <: Real })
  @. model(x, p) = p[1]*x+p[2]
  p0 = [-2.6, 1e1]
  ([coef(curve_fit(model, xydata[k-3:k+3,1], xydata[k-3:k+3,2], p0))[1] for k=4:size(xydata,1)-3],[stderror(curve_fit(model, xydata[k-3:k+3,1], xydata[k-3:k+3,2], p0))[1] for k=4:size(xydata,1)-3])
end
function inter_fun(x::Array{T,1} where {T<:Real},y::Array{T,1} where {T<:Real})
  fun(i) = exp(extrapolate(interpolate((log.(x),), log.(y), Gridded(Linear())), Line())(log.(i)))
  fun
end
#####################################
function local_spec(spec::Dict{String,Particle},fname::String)
       param=get_param(fname)
   result = zero_(spec)
    for name in keys(spec)
      if name=="Hydrogen_1"
         result[name] = local_particle(1,1,1.06e6,param["nu"],param)
      elseif name=="Oxygen_16"
       result[name] = local_particle(16,8,param["ab_o"],param["nu_cno"],param)
      elseif    name=="Nitrogen_14"
         result[name] = local_particle(14,7,param["ab_n"],param["nu_cno"],param)
      elseif    name=="Carbon_12"
         result[name] = local_particle(12,6,param["ab_c"],param["nu_cno"],param)
      elseif    name=="Neon_20"
         result[name] = local_particle(20,10,param["ab_ne"],param["nu_nemgsi"],param)
      elseif    name=="Neon_22"
         result[name] = local_particle(22,10,param["ab_ne2"],param["nu_nemgsi"],param)
      elseif    name=="Magnesium_24"
         result[name] = local_particle(24,12,param["ab_mg"],param["nu_nemgsi"],param)
      elseif    name=="Silicon_28"
         result[name] = local_particle(28,14,param["ab_si"],param["nu_nemgsi"],param)
      elseif    name=="Sodium_23"
         result[name] = local_particle(23,11,param["ab_na"],param["nu_nemgsi"],param)
      elseif    name=="Aluminium_27"
         result[name] = local_particle(27,13,param["ab_al"],param["nu_nemgsi"],param)
      elseif    name=="Iron_56"
         result[name] = local_particle(56,26,param["ab_fe"],param["nu_fe"],param)
      else
      result[name] = local_particle(spec[name].A,spec[name].Z,0,0,param)
      end
    end
    return result
end
function local_particle(A::Int,Z::Int,abun::Real,nu::Real,param::Dict{String,Float64})
  ene=[2* 1.2^i*1e-3 for i=0:110]
  VELOCITY_LIGHT =2.99792458e8;PC=3.08567758149e16;YEAR=3.15569252e7;m0=0.9382;
  l=param["l"]*PC;t=param["t_inj"]*YEAR;D_0=param["D_0"]*1e-4;delta=param["delta"]
  E_nucleon = ene.+m0
  E         = A*E_nucleon
  pMc       =@. sqrt(E^2-m0^2*A^2)
  rigidity  = pMc/Z
  beta      = pMc./E
  D=@.D_0* beta*(rigidity/4).^delta
  if ( (l-VELOCITY_LIGHT*t)>0 && param["t_inj2"]==0)
   return 0
  end
    idndr= Array{Float64}(undef, length(ene))*0   
    idndr+=-q_inj(A,Z,ene,abun,nu,param["r_cut"]).*(q_fra(A,Z,ene,param)*param["tau"]*YEAR .-1) 
    idndr+=q_sec(A,Z,ene,param)*param["tau"]*YEAR
     idndr=[ idndr[i]<0 ? 0 : idndr[i] for i in 1:length(ene)]
    idndr=idndr.*param["prot_norm"].*beta*VELOCITY_LIGHT/4pi
    idndr=idndr.* (param["t_inj2"]==0 ? exp.(-l^2 ./(4*D*t)).*(4pi*D*t).^(-3/2) : contine(l,D,D_0,param["t_inj"],param["t_inj2"]) )
    tmp = Particle(Array{Real,1}(), rigidity, idndr, ene, A, Z)
    count_Ekin(tmp)
   return tmp
end
function contine(l::Real,K_diff::Array{T,1} where {T<:Real},D0_xx::Real,t1_S::Real,t2_S::Real)
  n      = 2*200
  VELOCITY_LIGHT =2.99792458e8;YEAR=3.15569252e7
  lnT_u  = log(t2_S)
  lnT_l  = log(t1_S)
  dlnT   = (lnT_u -lnT_l)/n
  lnT    = lnT_l
  let prop   = Array{Float64}(undef, length(K_diff))*0   
  for i in 0:n
   t = exp(lnT)
    if(i == 0 || i == n)
     weight_SIMPSON = 1/3
    else
     weight_SIMPSON = (1+i%2)*2/3
    end    
     test = (VELOCITY_LIGHT*t*YEAR -l) < 0. ? 0. : 1.
    if test == 0
      prop .+= 0
    else
     prop += exp.(-l^2 ./(4*K_diff*t*YEAR)).*(4pi*K_diff/D0_xx).^(-3/2) /sqrt(t)*dlnT*weight_SIMPSON
    end
    lnT+= dlnT
  end
  prop  ./= (D0_xx^(3/2)*sqrt(YEAR))
  return prop
  end
end
function q_inj(A::Int,Z::Int,ene::Array{T,1} where {T<:Real},abun::Real,ν::Real,rc::Real)
  m0=0.9382
  E_nucleon = ene.+m0
  E         = A*E_nucleon
  pMc       =@. sqrt(E^2-m0^2A^2)
  rigidity  = pMc/Z
  beta      = pMc./E
  return @.rigidity^(-ν)*exp(-rigidity/rc)*abun/1.06e6
end
function q_fra(A::Int,Z::Int,ene::Array{T,1} where {T<:Real},param::Dict{String,Float64})
 DEN_H=0.9e6;MBARN=1e-31; 
   m0=0.9382;YEAR=3.15569252e7;VELOCITY_LIGHT =2.99792458e8
  E_nucleon = ene.+m0
  E         = A*E_nucleon
  pMc       =@. sqrt(E^2-m0^2A^2)
  beta      = pMc./E
  CS=get_cs("cs.dat")
  let sec= Array{Float64}(undef, length(ene))*0   
    if Z==6
      sec +=  CS["sigma_C12_total"] 
    elseif Z==7
      sec +=  CS["sigma_N14_total"] 
    elseif Z==8
      sec +=  CS["sigma_O16_total"] 
    elseif (Z==10 && A==20)
      sec +=  CS["sigma_Ne20_total"]
    elseif (Z==10 && A==22)
      sec +=  CS["sigma_Ne22_total"]
    elseif Z==11
      sec +=  CS["sigma_Na23_total"]  
    elseif Z==12
      sec +=  CS["sigma_Mg24_total"] 
    elseif Z==13
      sec +=  CS["sigma_Al27_total"] 
    elseif Z==14
      sec +=  CS["sigma_Si28_total"] 
    elseif Z==26
      sec +=  CS["sigma_Fe56_total"]
    end
   return sec*DEN_H*MBARN .*beta*VELOCITY_LIGHT
  end
end
function q_sec(A::Int,Z::Int,ene::Array{T,1} where {T<:Real},param::Dict{String,Float64})
 DEN_H=0.9e6;MBARN=1e-31; 
   m0=0.9382;YEAR=3.15569252e7;VELOCITY_LIGHT =2.99792458e8
  E_nucleon = ene.+m0
  E         = A*E_nucleon
  pMc       =@. sqrt(E^2-m0^2A^2)
  beta      = pMc./E
  CS=get_cs("cs.dat")
  let sec= Array{Float64}(undef, length(ene))*0
   if (Z==5 && A==10)
    name= "B10" 
    sec += CS["sigma_C12_$name"] .*q_inj(12,6,ene,param["ab_c"],param["nu_cno"],param["r_cut"])
    sec += CS["sigma_N14_$name"] .*q_inj(14,7,ene,param["ab_n"],param["nu_cno"],param["r_cut"])
    sec +=  CS["sigma_O16_$name"] .*q_inj(16,8,ene,param["ab_o"],param["nu_cno"],param["r_cut"])
    sec += CS["sigma_Ne20_$name"] .*q_inj(20,10,ene,param["ab_ne"],param["nu_nemgsi"],param["r_cut"])
    sec += CS["sigma_Ne22_$name"] .*q_inj(22,10,ene,param["ab_ne2"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Mg24_$name"] .*q_inj(24,12,ene,param["ab_mg"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Si28_$name"] .*q_inj(28,14,ene,param["ab_si"],param["nu_nemgsi"],param["r_cut"])
        sec +=  CS["sigma_Na23_$name"].*q_inj(23,11,ene,param["ab_na"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
   elseif (Z==5 && A==11) 
    name= "B11" 
    sec += CS["sigma_C12_$name"].*q_inj(12,6,ene,param["ab_c"],param["nu_cno"],param["r_cut"])
    sec += CS["sigma_N14_$name"] .*q_inj(14,7,ene,param["ab_n"],param["nu_cno"],param["r_cut"])
    sec +=  CS["sigma_O16_$name"] .*q_inj(16,8,ene,param["ab_o"],param["nu_cno"],param["r_cut"])
    sec += CS["sigma_Ne20_$name"] .*q_inj(20,10,ene,param["ab_ne"],param["nu_nemgsi"],param["r_cut"])
    sec += CS["sigma_Ne22_$name"].*q_inj(22,10,ene,param["ab_ne2"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Mg24_$name"].*q_inj(24,12,ene,param["ab_mg"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Si28_$name"].*q_inj(28,14,ene,param["ab_si"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Na23_$name"].*q_inj(23,11,ene,param["ab_na"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
   elseif (Z==9 && A==19)
    name= "F19" 
    sec += CS["sigma_Ne20_$name"].*q_inj(20,10,ene,param["ab_ne"],param["nu_nemgsi"],param["r_cut"])
    sec +=CS["sigma_Ne22_$name"].*q_inj(22,10,ene,param["ab_ne2"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Mg24_$name"].*q_inj(24,12,ene,param["ab_mg"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Si28_$name"].*q_inj(28,14,ene,param["ab_si"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Na23_$name"].*q_inj(23,11,ene,param["ab_na"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Fe56_$name"] .*q_inj(56,26,ene,param["ab_fe"],param["nu_fe"],param["r_cut"])
   elseif (Z==7 && A==14)
    name= "N14" 
    sec +=  CS["sigma_O16_$name"] .*q_inj(16,8,ene,param["ab_o"],param["nu_cno"],param["r_cut"])
    sec += CS["sigma_Ne20_$name"] .*q_inj(20,10,ene,param["ab_ne"],param["nu_nemgsi"],param["r_cut"])
    sec += CS["sigma_Ne22_$name"] .*q_inj(22,10,ene,param["ab_ne2"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Mg24_$name"] .*q_inj(24,12,ene,param["ab_mg"],param["nu_nemgsi"],param["r_cut"])
        sec +=  CS["sigma_Na23_$name"].*q_inj(23,11,ene,param["ab_na"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
   elseif (Z==7 && A==15)
    name= "N15" 
    sec +=  CS["sigma_O16_$name"] .*q_inj(16,8,ene,param["ab_o"],param["nu_cno"],param["r_cut"])
    sec += CS["sigma_Ne20_$name"] .*q_inj(20,10,ene,param["ab_ne"],param["nu_nemgsi"],param["r_cut"])
    sec += CS["sigma_Ne22_$name"] .*q_inj(22,10,ene,param["ab_ne2"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Mg24_$name"] .*q_inj(24,12,ene,param["ab_mg"],param["nu_nemgsi"],param["r_cut"])
        sec +=  CS["sigma_Na23_$name"].*q_inj(23,11,ene,param["ab_na"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
       elseif (Z==6 && A==12)
    name= "C12" 
        sec += CS["sigma_N14_$name"] .*q_inj(14,7,ene,param["ab_n"],param["nu_cno"],param["r_cut"])
    sec +=  CS["sigma_O16_$name"] .*q_inj(16,8,ene,param["ab_o"],param["nu_cno"],param["r_cut"])
    sec += CS["sigma_Ne20_$name"] .*q_inj(20,10,ene,param["ab_ne"],param["nu_nemgsi"],param["r_cut"])
    sec += CS["sigma_Ne22_$name"] .*q_inj(22,10,ene,param["ab_ne2"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Mg24_$name"] .*q_inj(24,12,ene,param["ab_mg"],param["nu_nemgsi"],param["r_cut"])
        sec +=  CS["sigma_Na23_$name"].*q_inj(23,11,ene,param["ab_na"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
       elseif (Z==6 && A==13)
    name= "C13" 
           sec += CS["sigma_N14_$name"] .*q_inj(14,7,ene,param["ab_n"],param["nu_cno"],param["r_cut"])
    sec +=  CS["sigma_O16_$name"] .*q_inj(16,8,ene,param["ab_o"],param["nu_cno"],param["r_cut"])
    sec += CS["sigma_Ne20_$name"] .*q_inj(20,10,ene,param["ab_ne"],param["nu_nemgsi"],param["r_cut"])
    sec += CS["sigma_Ne22_$name"] .*q_inj(22,10,ene,param["ab_ne2"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Mg24_$name"] .*q_inj(24,12,ene,param["ab_mg"],param["nu_nemgsi"],param["r_cut"])
        sec +=  CS["sigma_Na23_$name"].*q_inj(23,11,ene,param["ab_na"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
            elseif (Z==8 && A==16)
    name= "O16" 
    sec += CS["sigma_Ne20_$name"] .*q_inj(20,10,ene,param["ab_ne"],param["nu_nemgsi"],param["r_cut"])
    sec += CS["sigma_Ne22_$name"] .*q_inj(22,10,ene,param["ab_ne2"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Mg24_$name"] .*q_inj(24,12,ene,param["ab_mg"],param["nu_nemgsi"],param["r_cut"])
        sec +=  CS["sigma_Na23_$name"].*q_inj(23,11,ene,param["ab_na"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
                elseif (Z==8 && A==17)
    name= "O17" 
    sec += CS["sigma_Ne20_$name"] .*q_inj(20,10,ene,param["ab_ne"],param["nu_nemgsi"],param["r_cut"])
    sec += CS["sigma_Ne22_$name"] .*q_inj(22,10,ene,param["ab_ne2"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Mg24_$name"] .*q_inj(24,12,ene,param["ab_mg"],param["nu_nemgsi"],param["r_cut"])
        sec +=  CS["sigma_Na23_$name"].*q_inj(23,11,ene,param["ab_na"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
                elseif (Z==8 && A==18)
    name= "O18" 
    sec += CS["sigma_Ne20_$name"] .*q_inj(20,10,ene,param["ab_ne"],param["nu_nemgsi"],param["r_cut"])
    sec += CS["sigma_Ne22_$name"] .*q_inj(22,10,ene,param["ab_ne2"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Mg24_$name"] .*q_inj(24,12,ene,param["ab_mg"],param["nu_nemgsi"],param["r_cut"])
        sec +=  CS["sigma_Na23_$name"].*q_inj(23,11,ene,param["ab_na"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
                elseif (Z==10 && A==20)
    name= "Ne20" 
    sec +=  CS["sigma_Mg24_$name"].*q_inj(24,12,ene,param["ab_mg"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Si28_$name"].*q_inj(28,14,ene,param["ab_si"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Na23_$name"].*q_inj(23,11,ene,param["ab_na"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Fe56_$name"] .*q_inj(56,26,ene,param["ab_fe"],param["nu_fe"],param["r_cut"])
    elseif (Z==10 && A==21)
    name= "Ne21" 
    sec +=  CS["sigma_Mg24_$name"].*q_inj(24,12,ene,param["ab_mg"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Si28_$name"].*q_inj(28,14,ene,param["ab_si"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Na23_$name"].*q_inj(23,11,ene,param["ab_na"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Fe56_$name"] .*q_inj(56,26,ene,param["ab_fe"],param["nu_fe"],param["r_cut"])
                    elseif (Z==10 && A==22)
    name= "Ne22" 
    sec +=  CS["sigma_Mg24_$name"].*q_inj(24,12,ene,param["ab_mg"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Si28_$name"].*q_inj(28,14,ene,param["ab_si"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Na23_$name"].*q_inj(23,11,ene,param["ab_na"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Fe56_$name"] .*q_inj(56,26,ene,param["ab_fe"],param["nu_fe"],param["r_cut"])
                    elseif (Z==11 && A==23)
    name= "Na23" 
    sec +=  CS["sigma_Mg24_$name"].*q_inj(24,12,ene,param["ab_mg"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Si28_$name"].*q_inj(28,14,ene,param["ab_si"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Fe56_$name"] .*q_inj(56,26,ene,param["ab_fe"],param["nu_fe"],param["r_cut"])
                    elseif (Z==12 && A==24)
    name= "Mg24" 
    sec +=  CS["sigma_Si28_$name"].*q_inj(28,14,ene,param["ab_si"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Fe56_$name"] .*q_inj(56,26,ene,param["ab_fe"],param["nu_fe"],param["r_cut"])
                        elseif (Z==12 && A==25)
    name= "Mg25" 
    sec +=  CS["sigma_Si28_$name"].*q_inj(28,14,ene,param["ab_si"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Fe56_$name"] .*q_inj(56,26,ene,param["ab_fe"],param["nu_fe"],param["r_cut"])
                        elseif (Z==12 && A==26)
    name= "Mg26" 
    sec +=  CS["sigma_Si28_$name"].*q_inj(28,14,ene,param["ab_si"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Al27_$name"] .*q_inj(27,13,ene,param["ab_al"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Fe56_$name"] .*q_inj(56,26,ene,param["ab_fe"],param["nu_fe"],param["r_cut"])
                        elseif (Z==13 && A==26)
    name= "Al26" 
    sec +=  CS["sigma_Si28_$name"].*q_inj(28,14,ene,param["ab_si"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Fe56_$name"] .*q_inj(56,26,ene,param["ab_fe"],param["nu_fe"],param["r_cut"])
                            elseif (Z==13 && A==27)
    name= "Al27" 
    sec +=  CS["sigma_Si28_$name"].*q_inj(28,14,ene,param["ab_si"],param["nu_nemgsi"],param["r_cut"])
    sec +=  CS["sigma_Fe56_$name"] .*q_inj(56,26,ene,param["ab_fe"],param["nu_fe"],param["r_cut"])
                            elseif (Z==14 && A==28)
    name= "Si28" 
    sec +=  CS["sigma_Fe56_$name"] .*q_inj(56,26,ene,param["ab_fe"],param["nu_fe"],param["r_cut"])
                                elseif (Z==14 && A==29)
    name= "Si29" 
    sec +=  CS["sigma_Fe56_$name"] .*q_inj(56,26,ene,param["ab_fe"],param["nu_fe"],param["r_cut"])
                                elseif (Z==14 && A==30)
    name= "Si30" 
    sec +=  CS["sigma_Fe56_$name"] .*q_inj(56,26,ene,param["ab_fe"],param["nu_fe"],param["r_cut"])
   end
   if (Z>8 && Z<26)
    sec*=1.1                #considering higher Z contribution
   end
   return sec*DEN_H*MBARN .*beta*VELOCITY_LIGHT
  end
end
function get_param(fname::String)
  result = Dict{String, Float64}()
  key = ""
  open(fname) do file
    while !eof(file)
      line = readline(file)
      for i=1:length(line)
        if line[i] == '='
         key = line[1:i-1]
       #  result[key] = Float64()
         result[key] = parse(Float64, line[i+1:length(line)])
        end
      end
    end
  end
  result
end
function get_cs(fname::String)
  basedir = dirname(@__FILE__)
  result = Dict{String, Array{Float64,1}}()
  key = ""
  open("$basedir/$fname") do file
    while !eof(file)
      line = readline(file)
      if line[1] == '#'
        key = line[2:length(line)]
        result[key] = Array{Float64,1}(undef, 0)
      else
        if (key != "")
          lvec = map(x->parse(Float64, x), split(line))
          result[key] = lvec
        end
      end
    end
  end
  result
end
###################################################
"""
  use pure=1 to turn off experiments
  use pure=-1 for forcing experiment marker type and color
"""
function purety(A::Int;col::Symbol = :yellow,mar_index::Int = 1)
 global pure= A
 global color=col
 global marker_i=mar_index
end
function mod_model(name::String)
 global modulation_model= name
end

purety(0)
mod_model("FFA")
###############
end # module
