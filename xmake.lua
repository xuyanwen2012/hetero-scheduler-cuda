add_rules("mode.debug", "mode.release")

add_requires("cuda", {system = true})

add_requires("libomp", {optional = true})

set_languages("c++20")
set_warnings("all")

if is_mode("debug") then
    set_symbols("debug")
    set_optimize("none")
end

if is_mode("release") then
    set_symbols("hidden")
    set_optimize("fastest")
end

-- target("bandwidth")
--     set_kind("binary")
--     add_includedirs("include")
--     add_files("src/test_bandwidth.cu")
--     add_cugencodes("native")

target("main")
    set_kind("binary")
    set_default(true)
    add_includedirs("include")
    add_headerfiles("include/*.h", "include/**/*.h")
    add_files("src/main.cu")

    add_packages("libomp")
    -- add_packages("cuda", "cutlass")

    add_cugencodes("native")
    add_cugencodes("compute_75")


