add_executable(openpgl_cuda_fit
    main.cpp
)

target_link_libraries(openpgl_cuda_fit PRIVATE openpgl)
INSTALL(TARGETS openpgl_cuda_fit RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
