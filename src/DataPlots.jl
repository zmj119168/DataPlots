VERSION < v"0.1.0" && __precompile__()

module DataPlots

export plot_BC
export plot_proton
export plot_pbar
export modulation

using Plots
using Interpolations

"""
    modulation(ene::Array{T,1} where {T<:Real}, flux::Array{T,1} where {T<:Real};
               A::Int = 1, Z::Int = 1, phi::Real = 0)

Doing solar modulation for specified spectrum; ene in unit [GeV]; flux is the nucleon flux

# Arguments
* `A`,`Z`:  the A and Z of the particle
* `phi`:    modulation potential [unit: GV]
"""
function modulation(ene::Array{T,1} where {T<:Real}, flux::Array{T,1} where {T<:Real}; A::Int = 1, Z::Int = 1, phi::Real = 0)
  phi_ = phi * abs(Z) / A
  m0 = A == 0 ? 0.511e-3 : 0.9382

  logene = log.(ene)
  itpspec = interpolate((range(logene[1],last(logene),length=length(logene)),), log.(flux), Gridded(Linear()))
  spec = extrapolate(itpspec, Line())

  (ene, map(e-> e * (e + 2 * m0) / ( (e + phi_) * (e + phi_ + 2 * m0)) * exp(spec(log(e + phi_))), ene))
end

function get_data(fname::String; index::Real = 0.0, norm::Real = 1.0)
  basedir = dirname(@__FILE__)
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

function plot_data(data::Array{T,2} where { T <: Real })
  plot(data[:,1], data[:,2];yerror=data[:,3], linewidth=0, marker=:dot, label="")
end

"""
    plot_BC(spectra::Array{Dict{String,Array{Float64,1}},1}, label::Array{String,2})

    Ploting the B/C ratio of given spectra in comparison with the data
"""
function plot_BC(spectra::Array{Dict{String,Array{Float64,1}},1}, label::Array{String, 2})
  data = get_data("bcratio.dat")
  plot_data(data["AMS02(2011/05-2016/05)"])
  plot!(xaxis=:log, xlabel="Ekin[GeV]")

  bc = map(spec->(spec["Boron_10"] + spec["Boron_11"]) ./ (spec["Carbon_12"] + spec["Carbon_13"]), spectra)

  plot!(spectra[1]["eaxis"], bc; label = label)
end

"""
    plot_proton(spectra::Array{Dict{String,Array{Float64,1}},1}, label::Array{String,2})

    Ploting the proton ratio of given spectra in comparison with the data
"""
function plot_proton(spectra::Array{Dict{String,Array{Float64,1}},1}, label::Array{String, 2})
  data = get_data("proton.dat"; norm=1e-4)
  plot_data(data["AMS2015(2011/05-2013/11)"])
  plot!(xaxis=:log, yaxis=:log)

  proton = map(spec-> (spec["Hydrogen_1"] + spec["Hydrogen_2"]) .* (spec["eaxis"] .^ 2.7), spectra)
  plot!(spectra[1]["eaxis"], proton; label = label)
end

"""
    plot_pbar(spectra::Array{Dict{String,Array{Float64,1}},1}, label::Array{String,2})

    Ploting the proton ratio of given spectra in comparison with the data
"""
function plot_pbar(spectra::Array{Dict{String,Array{Float64,1}},1}, label::Array{String, 2})
  data = get_data("pbar.dat"; index=-2, norm=1e-4)
  plot_data(data["AMS2016nonformal(0000/00)"])
  plot!(xaxis=:log, yaxis=:log)

  #pbar = map(spec-> (spec["secondary_antiprotons"] + spec["tertiary_antiprotons"]) .* (spec["eaxis"] .^ 2), spectra)
  pbar = map(spec-> max.(spec["DM_antiprotons"] .* (spec["eaxis"] .^ 2), 1e-7), spectra)
  print(pbar)
  plot!(spectra[1]["eaxis"], pbar; label = label)
end

end # module
