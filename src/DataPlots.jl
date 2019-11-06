VERSION < v"0.1.0" && __precompile__()

module DataPlots

export plot_BC
export plot_proton
export plot_pbar
export modulation
export +

using Plots
using Interpolations
using FITSUtils
using Printf
using FITSIO

"""
    modulation(ene::Array{T,1} where {T<:Real}, flux::Array{T,1} where {T<:Real};
               A::Int = 1, Z::Int = 1, phi::Real = 0)

Doing solar modulation for specified spectrum; ene in unit [GeV]; flux is the nucleon flux

# Arguments
* `A`,`Z`:  the A and Z of the particle
* `phi`:    modulation potential [unit: GV]
"""
function modulation(ene::Array{T,1} where {T<:Real}, flux::Array{T,1} where {T<:Real}; A::Int = 1, Z::Int = 1,phi::Real = 0)
  phi_ = phi * abs(Z) / A
  m0 = A == 0 ? 0.511e-3 : 0.9382

  logene = log.(ene)
  itpspec = interpolate((range(logene[1],last(logene),length=length(logene)),), log.(flux), Gridded(Linear()))
  spec = extrapolate(itpspec, Line())

  (ene, map(e-> e * (e + 2 * m0) / ( (e + phi_) * (e + phi_ + 2 * m0)) * exp(spec(log(e + phi_))), ene)) 
end

"""
    modulation(particle::Particle, phi::Real)

Doing solar modulation for specified spectrum inside the Particle structure

# Arguments
* `phi`:    modulation potential [unit: GV]
"""
function modulation(particle::Particle, phi::Real)
  particle.Ekin, particle.dNdE = modulation(particle.Ekin, particle.dNdE; A=particle.A, Z=particle.Z, phi=phi)
  count_R(particle)
end

"""
    dict_modulation(spec::Dict{String,Particle}, phi::Real = 0)

Doing solar modulation for a spectra dict of many Particles produced by FITSUtils

# Arguments
* `phi`:    modulation potential [unit: GV]
"""
function dict_modulation(spec::Dict{String,Particle}, phi::Real = 0)
  phi == 0 && return spec

  mod_spec = Dict{String,Particle}()
  for k in keys(spec)
    mod_spec[k] = modulation(copy(spec[k]), phi)
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
  plot(data[:,1], data[:,2]; yerror=data[:,3], linewidth=0, marker=:dot, label=label)
end

function plot_data!(data::Array{T,2} where { T <: Real },label::String)
  plot!(data[:,1], data[:,2]; yerror=data[:,3], linewidth=0, marker=:dot, label=label)
end

function plot_comparison(plot_func, spectra::Array{Dict{String,Particle},1}, label::Array{String,2};
                         phi::Real = 0,
                         data::Array{String,1} = [],
                         datafile::String = "", index::Real = 0, norm::Real = 1,
                         xscale::Symbol = :log10, yscale::Symbol = :none,
                         ylabel::String = "")
  label[1]*=",phi="*string(phi)
  whole_ekin = true
  whole_rigidity = false
  if length(data) != 0
    pdata = get_data(datafile; index=index, norm=norm)

    if !mapreduce(k->haskey(pdata, k), &, data)
      data_keys = keys(pdata)
      throw("The specified data not found, note that the available data in file $datafile are:\n$data_keys")
    end

    for k in data
      plot_data!(pdata[k], k)
    end

    whole_rigidity = mapreduce(k->occursin("rigidity", k), &, data)
    whole_ekin = mapreduce(k->!occursin("rigidity", k), &, data)
  end

  if !whole_rigidity && !whole_ekin
    throw("The specified data should be all in rigidity or all in Ekin")
  end
  plot!(xscale=xscale, xlabel=whole_ekin ? "Ekin[GeV]" : "R[GV]", yscale=yscale, ylabel=whole_ekin ? ylabel : replace(replace(ylabel,"e"=>""),"E"=>"R"))

  mod_spectra = map(spec->dict_modulation(spec,phi), spectra)
  for i in 1:length(mod_spectra)
    ptc = plot_func(mod_spectra[i])
    ene, flux = whole_ekin ? (ptc.Ekin, ptc.dNdE) : (ptc.R, ptc.dNdR)
    plot!(ene, flux; label = i<=length(label) ? label[1,i] : "")
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

Base.:+(a::Particle, b::Particle) = op_particle(a, b, +)
Base.:-(a::Particle, b::Particle) = op_particle(a, b, -)
Base.:*(a::Particle, b::Particle) = op_particle(a, b, *)
Base.:/(a::Particle, b::Particle) = op_particle(a, b, /)

function Base.:*(a::Particle, n::Real)
  a.dNdE .*= n
  a.dNdR .*= n
  a
end

function rescale(a::Particle, index::Real)
  a.dNdE = @. a.dNdE * a.Ekin^index
  a.dNdR = @. a.dNdR * a.R^index
  a
end

"""
    plot_BC(spectra::Array{Dict{String,Particle},1}, label::Array{String,2};
            phi::Real = 0, data::Array{String,1})  

    Ploting the B/C ratio of given spectra in comparison with the data

# Arguments
* `phi`:     modulation potential [unit: GV]
* `data`:    The dataset to plot
"""
function plot_BC(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0)); phi::Real = 0, data::Array{String,1}=["AMS02(2011/05-2016/05)"])
  plot_comparison(spec-> (spec["Boron_10"] + spec["Boron_11"]) / (spec["Carbon_12"] + spec["Carbon_13"]),
                  spectra, label; phi=phi, data=data, datafile="bcratio.dat", ylabel="B/C")
end

"""
    plot_proton(spectra::Array{Dict{String,Particle},1}, label::Array{String,2};
            phi::Real = 0, data::Array{String,1})  

    Ploting the proton flux of given spectra in comparison with the data

# Arguments
* `phi`:     modulation potential [unit: GV]
* `data`:    The dataset to plot
"""
function plot_proton(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0)); phi::Real = 0, data=["AMS2015(2011/05-2013/11)"])
  plot_comparison(spec -> rescale(spec["Hydrogen_1"] + spec["Hydrogen_2"], 2.7) * 1e4,
                  spectra, label; phi=phi, data=data, datafile="proton.dat", yscale=:log, ylabel="\$E^{2.7}dN/dE [GeV^{2.7}(m^{2}*sr*s*GeV)^{-1}]\$")
end


"""
    plot_pbar(spectra::Array{Dict{String,Particle},1}, label::Array{String,2};
            phi::Real = 0, data::Array{String,1})  

    Ploting the antiproton flux of given spectra in comparison with the data

# Arguments
* `phi`:     modulation potential [unit: GV]
* `data`:    The dataset to plot
"""
function plot_pbar(spectra::Array{Dict{String,Particle},1}, label::Array{String,2} = Array{String,2}(undef, (0,0)); phi::Real = 0, data=["AMS2015(2011/05-2013/11)"])
  plot_comparison(spec -> rescale(spec["secondary_antiprotons"] + spec["tertiary_antiprotons"], 2.0) * 1e4,
                  spectra, label; phi=phi, data=data, datafile="pbar.dat", yscale=:log, ylabel="\$E^{2}dN/dE [GeV^{2}(m^{2}*sr*s*GeV)^{-1}]\$")
 #plot_comparison(spec -> rescale(spec["DM_antiprotons"], 2.0) * 1e-3,
 #                spectra, label; phi=phi, data=data, datafile="pbar.dat", yscale=:log, ylabel="\$E^{2}dN/dE [GeV^{2}(m^{2}*sr*s*GeV)^{-1}]\$")
end
end # module
