struct Tag{O <: Unsigned}
    tag::UInt16
    datatype::DataType
    count::O
    data::Vector{UInt8}
    loaded::Bool
end

function load(tf::TiffFile{O}, t::Tag{O}) where {O}
    (t.loaded) && return t

    loc = first(reinterpret(O, getfield(t, :data)))

    data = Vector{UInt8}(undef, bytes(t.datatype)*t.count)

    pos = position(tf.io)
    seek(tf, loc)
    read!(tf, data)

    # if this datatype is comprised of multiple bytes and this file needs to be
    # bitswapped then we'll need to reverse the byte order inside each datatype
    # unit 
    if tf.need_bswap && bytes(t.datatype) >= 2
        reverse!(data)
        data .= Array(reinterpret(UInt8, Array{t.datatype}(reverse(reinterpret(t.datatype, data)))))
    end
    seek(tf, pos)

    Tag(t.tag, t.datatype, t.count, data, true)
end

function Base.getproperty(t::Tag{O}, sym::Symbol) where {O}
    (sym != :data) && return getfield(t, sym)

    if !t.loaded
        error("This tag has remote data and it hasn't been loaded yet. Call `load!` first")
    end

    T = t.datatype 
    T = T == Any ? UInt8 : T

    data = getfield(t, sym)
    if T == UInt8
        converted = data
    elseif isbitstype(T)
        converted = reinterpret(t.datatype, data)
    elseif T == String
        converted = String(data)
    else
        error("Unexpected tag type")
    end

    converted
end

bytes(x::Type) = sizeof(x)
bytes(::Type{Any}) = 1
bytes(::Type{String}) = 1

function Base.read(tf::TiffFile, ::Type{Tag{O}}) where O <: Unsigned
    tag = read(tf, UInt16)
    datatype = read(tf, UInt16)
    count = read(tf, O)
    data = Vector{UInt8}(undef, sizeof(O))
    read!(tf, data)

    T = Any
    if datatype in keys(tiff_to_julian)
        T = tiff_to_julian[datatype]
    end

    nbytes = bytes(T) * count
    if nbytes <= sizeof(O)
        if tf.need_bswap
            reverse!(view(data, 1:nbytes))
        end
        Tag(tag, T, count, data, true)
    else
        (tf.need_bswap) && reverse!(data)
        Tag(tag, T, count, data, false)
    end
end

function Base.show(io::IO, t::Tag{O}) where {O}
    print("TIFF.Tag{$O}(")
    try
        print(TiffTag(t.tag), ", ")
    catch
        print("UNKNOWN($(Int(t.tag))), ")
    end
    print(t.datatype, ", ")
    print(Int(t.count), ", ")
    if t.tag == Int(COMPRESSION)
        print(CompressionType(first(t.data)))
    else
        if t.loaded
            if length(t.data) >= 1
                try
                    print(first(t.data))
                catch
                    println("%%%%%%%", length(t.data), "%%%%%%%%%")
                end
            else
                print("""\"\"""")
            end
        else
            print("***")
        end
    end
    print(")")
end

"""
    write(tf, t)

Write tag `t` to the tiff file `tf`. Returns `true` if the tag data fit
entirely in the IFD space and was written to disk. Otherwise it returns false.
"""
function Base.write(tf::TiffFile{O}, t::Tag{O}) where O <: Unsigned
    # if the data are too large to fit then we'll need to skip writing this tag
    # for now until we know the length of the entire IFD
    if t.loaded && length(t.data)*bytes(t.datatype) > sizeof(O)
        @info "This data will have to be remote"

        _writeblank(tf)
        return false
    end

    write(tf, t.tag)
    write(tf, julian_to_tiff[t.datatype])
    write(tf, t.count)
    nbytes = write(tf, getfield(t, :data))

    # write padding
    if nbytes < sizeof(O)
        write(tf, fill(0x00, sizeof(O) - nbytes))
    end
    true
end

function Base.write(tf::TiffFile, t::Tag{O}, offset) where {O <: Unsigned}
    write(tf, Tag(t.tag, t.datatype, t.count, Array(reinterpret(UInt8, [O(offset)])), false))
end

# Base.write(tf::TiffFile, t::Tag) = error("Tag offsets must agree with file offsets")

function _writeblank(tf::TiffFile{O}) where O
    write(tf, UInt32(0))
    write(tf, zero(O))
    write(tf, zero(O))
end

const tagfields = fieldnames(Tag)
function Base.:(==)(t1::Tag{O1}, t2::Tag{O2}) where {O1, O2}
    return O1 == O2 && getfield.(Ref(t1), tagfields) == getfield.(Ref(t2), tagfields)
end
