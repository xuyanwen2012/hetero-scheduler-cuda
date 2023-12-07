add_rules("mode.debug", "mode.release")

add_requires("cuda", {system = true})
add_requires("spdlog", "cli11")
add_requires("openmp")

set_languages("c++17")
set_warnings("all")

if is_mode("debug") then
    set_symbols("debug")
    set_optimize("none")
end

if is_mode("release") then
    set_symbols("hidden")
    set_optimize("fastest")
end


target("main")
    set_kind("binary")
    set_default(true)
    add_includedirs("include")
    add_headerfiles("include/*.hpp", "include/**/*.hpp")
    add_files("src/main.cpp")


    add_packages("cli11", "spdlog")
    add_packages("openmp")

    add_cugencodes("native")


