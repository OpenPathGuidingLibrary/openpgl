// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

/*
TODO:
- how to do float4, fancy packing tricks?
  flat int:32;
  float float:32
  struct Foo { int a, b; float c, d; };
  would be nice to load as a big float4...

- mechanism to not store fields that are easily recomputed...
  maybe the answer is to just do that--recompute only when needed--in the
  original struct!
*/

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

int line = 1;

#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#endif  // __GNUG__

const char *filename;

template <typename... Args>
static void error(const char *fmt, Args... args) {
    fprintf(stderr, "%s:%d: ", filename, line);
    fprintf(stderr, fmt, std::forward<Args>(args)...);
    exit(1);
}

#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif  // __GNUG__

struct OptionalString {
    OptionalString() = default;
    OptionalString(std::string s) : s(s), set(true) {}

    operator bool() const { return set; }
    operator std::string() const {
        assert(set);
        return s;
    }
    bool operator==(const char *str) const {
        assert(set);
        return s == str;
    }
    bool operator!=(const char *str) const {
        assert(set);
        return s != str;
    }

    std::string s;
    bool set = false;
};

struct Member {
    std::string type;
    bool isConst = false;
    bool isSOA = false;
    int numPointers = 0;

    std::string GetType() const {
        std::string s;
        if (isConst)
            s = "const ";
        s += type;
        for (int i = 0; i < numPointers; ++i)
            s += "*";
        return s;
    }

    bool IsSOA() const {
        return isSOA;
    }

    std::vector<std::string> names;
    std::vector<std::string> arraySizes;
};

struct SOA {
    std::string type;
    std::string templateType;
    std::vector<Member> members;
};

int main(int argc, char *argv[]) {
    if (argc != 2)
        error("usage: soac <soac filename>\n");

    // Read the file
    filename = argv[1];
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        error("%s: %s", filename, strerror(errno));
        return {};
    }
    std::string fileContents((std::istreambuf_iterator<char>(ifs)),
                             (std::istreambuf_iterator<char>()));
    int pos = 0;

    auto eof = [&]() { return pos == fileContents.size(); };
    auto getc = [&]() {
        assert(!eof());
        if (fileContents[pos] == '\n')
            ++line;
        return fileContents[pos++];
    };
    auto ungetc = [&]() {
        assert(pos > 0);
        --pos;
        if (fileContents[pos] == '\n')
            --line;
    };

    std::function<OptionalString(bool)> getToken;
    getToken = [&](bool eofOk) -> OptionalString {
        if (eof()) {
            if (eofOk)
                return OptionalString();
            else
                error("Premature end of file.\n");
        }

        // Skip whitespace
        while (true) {
            if (eof()) {
                if (eofOk)
                    return OptionalString();
                else
                    error("Premature end of file.\n");
            }

            char c = getc();
            if (!isspace(c)) {
                ungetc();
                break;
            }
        }

        assert(!eof());
        std::string s;
        s += getc();
        if (s[0] == '/' && !eof()) {
            if (getc() == '/') {
                // skip to EOL
                while (true) {
                    if (eof()) {
                        if (eofOk)
                            return OptionalString();
                        else
                            error("Premature end of file.\n");
                    }
                    if (getc() == '\n')
                        return getToken(eofOk);
                }
            } else
                ungetc();
        }
        if (!isalpha(s[0]) && s[0] != '_')
            return OptionalString(s);

        while (!eof()) {
            char c = getc();
            if (!isalnum(c) && c != '_') {
                // end of token
                ungetc();
                break;
            }
            s += c;
        }
        return OptionalString(s);
    };

    std::set<std::string> flatTypes, externSOA;
    auto isFlatType = [&](std::string type) {
        return flatTypes.find(type) != flatTypes.end();
    };

    // keep as a vector so that we can emit them in the order they were
    // defined.
    std::vector<SOA> soaTypes;

    auto soaTypeExists = [&](std::string type) {
        for (const auto &s : soaTypes)
            if (s.type == type)
                return true;
        return externSOA.find(type) != externSOA.end();
    };

    auto expect = [&](const char *str) {
        OptionalString tok = getToken(true);
        if (!tok)
            error("Premature end of file; expected \"%s\".\n", str);
        if (tok != str)
            error("Syntax error: expected \"%s\".\n", str);
    };

    while (true) {
        OptionalString os = getToken(true);
        if (!os)
            break;
        std::string tok = os.s;
        if (tok == "flat") {
            OptionalString typeTok = getToken(false);

            std::string type = typeTok;
            if (flatTypes.find(type) != flatTypes.end())
                error("%s flat type redeclared.\n", type.c_str());
            flatTypes.insert(type);

            expect(";");
        } else if (tok == "soa") {
            SOA soa;

            OptionalString typeTok = getToken(false);
            soa.type = (std::string)typeTok;
            if (!isalpha(soa.type[0]))
                error("%s: invalid type identifier.\n", soa.type.c_str());

            if (soaTypeExists(soa.type))
                error("%s: type redefined.\n", soa.type.c_str());

            OptionalString tok = getToken(false);
            if (tok == "<") {
                tok = getToken(false);
                soa.templateType = (std::string)tok;
                if (!isalpha(soa.templateType[0]))
                    error("%s: invalid type identifier.\n", soa.templateType.c_str());
                expect(">");
                expect("{");
            } else if (tok == ";") {
                externSOA.insert(soa.type);
                continue;
            } else if (tok != "{")
                error("Syntax error: expected \"{\".\n");

            while (true) {
                OptionalString tok = getToken(false);
                if (tok == "}")
                    break;

                Member member;
                member.type = (std::string)tok;
                // Hacks to parse things like const Foo *
                if (member.type == "const") {
                    member.isConst = true;
                    tok = getToken(false);
                    member.type = (std::string)tok;
                }
                member.isSOA = soaTypeExists(member.type);
                while (true) {
                    tok = getToken(false);
                    if (tok == "*")
                        ++member.numPointers;
                    else
                        break;
                }

                // Don't check the type if it's a pointer; we already know
                // how to SOA pointers..
                if (member.numPointers == 0 && member.type != soa.templateType &&
                    flatTypes.find(member.type) == flatTypes.end() &&
                    !soaTypeExists(member.type))
                    error("%s: undefined type\n", member.type.c_str());

                while (true) {
                    std::string memberName = tok;
                    member.names.push_back(memberName);
                    member.arraySizes.push_back("");  // assume no array for starters

                    tok = getToken(false);
                    if (tok == "[") {
                        std::string arraySize = "";
						while ((tok = getToken(false)) != "]") {
							arraySize += std::string(tok);
						}
						// just pass it through without interpretation
						member.arraySizes[member.arraySizes.size() - 1] =
							arraySize;
						//expect("]");
						tok = getToken(false);
                    }

                    if (tok == ";")
                        break;
                    else if (tok == ",")
                        tok = getToken(false);  // and go around again...
                }

                if (member.names.empty())
                    error("No members specified after type declaration.\n");
                soa.members.push_back(member);
            }
            expect(";");

            soaTypes.push_back(soa);
        } else
            error("%s: invalid token", tok.c_str());
    }

    // And now emit them...
    printf("// SOA definitions automatically generated by soac\n");
    printf("// DO NOT EDIT THIS FILE MANUALLY\n\n");
    printf("template <typename T> struct SOA;\n\n");
    for (const auto &soa : soaTypes) {
        if (!soa.templateType.empty())
            printf("template <typename %s> struct SOA<%s<%s>> {\n",
                   soa.templateType.c_str(), soa.type.c_str(), soa.templateType.c_str());
        else
            printf("template <> struct SOA<%s> {\n", soa.type.c_str());

        // Constructor
        printf("    SOA() = default;\n");
        printf("    SOA(int n, openpgl::gpu::Device *_device, bool _managed = false) : nAlloc(n), managed(_managed), device(_device) {\n");
        for (const auto &member : soa.members) {
            for (int i = 0; i < member.names.size(); ++i) {
                std::string name = member.names[i];
                if (!member.arraySizes[i].empty()) {
                    printf("        for (int i = 0; i < %s; ++i)\n",
                           member.arraySizes[i].c_str());
                    if (isFlatType(member.type) || member.numPointers > 0)
                        printf(
                            "            this->%s[i] = device->mallocArray<%s>(n, managed);\n",
                            name.c_str(), member.GetType().c_str());
                    else {
                        assert(member.isConst == false && member.numPointers == 0);
                        printf("        this->%s[i] = SOA<%s>(n, device);\n", name.c_str(),
                               member.type.c_str());
                    }
                } else {
                    if (isFlatType(member.type) || member.numPointers > 0)
                        printf("        this->%s = device->mallocArray<%s>(n, managed);\n",
                               name.c_str(), member.GetType().c_str());
                    else
                        printf("        this->%s = SOA<%s>(n, device);\n", name.c_str(),
                               member.type.c_str());
                }
            }
        }
        printf("        device->wait();\n");
        printf("    }\n");

        printf("    void Release() {\n");
        //printf("        std::cout << \"SOA()\" << std::endl;\n");
        printf("        if (device) {\n");
        printf("            device->wait();\n");
        for (const auto &member : soa.members) {
            for (int i = 0; i < member.names.size(); ++i) {
                std::string name = member.names[i];
                if (!member.arraySizes[i].empty()) {
                    printf("            for (int i = 0; i < %s; ++i) {\n",
                           member.arraySizes[i].c_str());
                    if (isFlatType(member.type) || member.numPointers > 0)
                        printf(
                            "                device->freeArray<%s>(this->%s[i]);\n",
                            member.GetType().c_str(), name.c_str());
                    else {
                        printf(
                            "                this->%s[i].Release();\n", name.c_str());
                    }
                    printf("            }\n");
                } else {
                    if (isFlatType(member.type) || member.numPointers > 0)
                        printf("            device->freeArray<%s>(this->%s);\n",
                               member.GetType().c_str(), name.c_str());
                    else
                        printf(
                            "                this->%s.Release();\n", name.c_str());
                }
            }
        }
        printf("            device->wait();\n");
        printf("        }\n");
        printf("    }\n");

        /**************************************************************/
		/*                  Modified to support resizing              */
		/**************************************************************/
		/*
        printf("    void resize(int n, Allocator alloc) { \n");
		for (const auto& member : soa.members) {
			for (int i = 0; i < member.names.size(); ++i) {
				std::string name = member.names[i];
				if (!member.arraySizes[i].empty()) {
					printf("        for (int i = 0; i < %s; ++i)\n",
						member.arraySizes[i].c_str());
					if (isFlatType(member.type) || member.numPointers > 0) {
						printf("            if (nAlloc) alloc.deallocate_object(this->%s[i]);\n", name.c_str());
						printf("            this->%s[i] = alloc.allocate_object<%s>(n);\n",
							name.c_str(), member.GetType().c_str());
					}
					else {
						assert(member.isConst == false && member.numPointers == 0);
						printf("        this->%s[i].resize(n, alloc);\n", name.c_str());
					}
				}
				else {
					if (isFlatType(member.type) || member.numPointers > 0) {
						printf("        if (nAlloc) alloc.deallocate_object(this->%s);\n", name.c_str());
						printf("        this->%s = alloc.allocate_object<%s>(n);\n",
							name.c_str(), member.GetType().c_str());
					}
					else
						printf("        this->%s.resize(n, alloc);\n", name.c_str());
				}
			}
		}
		printf("        nAlloc = n; \n");
		printf("    }\n");
        */
		/**************************************************************/
		/*                      Modification ended                    */
		/**************************************************************/
        printf("    SOA(const SOA& s) {\n");
        printf("        nAlloc = s.nAlloc;\n");
        printf("        device = s.device;\n");
        printf("        managed = s.managed;\n");
        
        for (const auto &member : soa.members) {
            for (int i = 0; i < member.names.size(); ++i) {
                std::string name = member.names[i];
                if (!member.arraySizes[i].empty()) {
                    printf("        for (int i = 0; i < %s; ++i)\n",
                           member.arraySizes[i].c_str());
                    printf("            this->%s[i] = s.%s[i];\n", name.c_str(),
                           name.c_str());
                } else {
                    printf("        this->%s = s.%s;\n", name.c_str(), name.c_str());
                }
            }
        }
        printf("    }\n");

        printf("    SOA &operator=(const SOA& s) {\n");
        printf("        nAlloc = s.nAlloc;\n");
        printf("        device = s.device;\n");
        printf("        managed = s.managed;\n");
        
        for (const auto &member : soa.members) {
            for (int i = 0; i < member.names.size(); ++i) {
                std::string name = member.names[i];
                if (!member.arraySizes[i].empty()) {
                    printf("        for (int i = 0; i < %s; ++i)\n",
                           member.arraySizes[i].c_str());
                    printf("            this->%s[i] = s.%s[i];\n", name.c_str(),
                           name.c_str());
                } else {
                    printf("        this->%s = s.%s;\n", name.c_str(), name.c_str());
                }
            }
        }
        printf("        return *this;\n");
        printf("    }\n");

        printf("    void serialize(std::ostream &os) const{\n");
        printf("        uint32_t n = this->nAlloc;\n");
        printf("        os.write(reinterpret_cast<const char *>(&n), sizeof(n));\n");
        printf("        os.write(reinterpret_cast<const char *>(&managed), sizeof(bool));\n");

        for (const auto &member : soa.members) {
            for (int i = 0; i < member.names.size(); ++i) {
                std::string name = member.names[i];
                printf("// %s is soa = %d\n", name.c_str(), (int)member.IsSOA());
                if (!member.arraySizes[i].empty()) {
                    if(!member.IsSOA()) {
                        printf("        %s* tmp_%s = new %s[(%s) * n];\n",member.GetType().c_str(), name.c_str(), member.GetType().c_str(), member.arraySizes[i].c_str());
                        printf("        for (int i = 0; i < %s; ++i) {\n",
                            member.arraySizes[i].c_str());
                    
                        printf("            device->memcpyArrayFromGPU<%s>(%s[i], &(tmp_%s[i*n]), n);\n", member.GetType().c_str(), name.c_str(), name.c_str());
                        printf("        }\n");
                        printf("        device->wait();\n");
                        printf("        os.write(reinterpret_cast<const char *>(tmp_%s), sizeof(%s) * ((%s)*n));\n", name.c_str(), member.GetType().c_str(), member.arraySizes[i].c_str());
                        printf("        delete[] tmp_%s;\n", name.c_str());
                    } else {
                        printf("        for (int i = 0; i < %s; ++i) {\n",
                            member.arraySizes[i].c_str());
                        printf("            %s[i].serialize(os);\n", name.c_str());
                        printf("        }\n");
                    }
                } else {
                    if(!member.IsSOA()) {
                        printf("        %s* tmp_%s = new %s[n];\n", member.GetType().c_str(), name.c_str(), member.GetType().c_str());
                        printf("        device->memcpyArrayFromGPU<%s>(%s, tmp_%s, n);\n", member.GetType().c_str(), name.c_str(), name.c_str());
                        printf("        device->wait();\n");
                        printf("        os.write(reinterpret_cast<const char *>(tmp_%s), sizeof(%s) * n);\n", name.c_str(), member.GetType().c_str());
                        printf("        delete[] tmp_%s;\n", name.c_str());
                    } else {
                        printf("        %s.serialize(os);\n", name.c_str());
                    }
                }
                
                printf("\n");
            }
        }        
        printf("        device->wait();\n");
        printf("    }\n");
        printf("\n");

        printf("    void deserialize(std::istream &is, openpgl::gpu::Device *_device) {\n");
        printf("        uint32_t n;\n");
        printf("        bool managed;\n");
        printf("        is.read(reinterpret_cast<char *>(&n), sizeof(n));\n");
        printf("        is.read(reinterpret_cast<char *>(&managed), sizeof(bool));\n");

        printf("        new (this) SOA<%s>(n, _device, managed);\n", soa.type.c_str());

        for (const auto &member : soa.members) {
            for (int i = 0; i < member.names.size(); ++i) {
                std::string name = member.names[i];
                printf("// %s is soa = %d\n", name.c_str(), (int)member.IsSOA());
                if (!member.arraySizes[i].empty()) {
                    if(!member.IsSOA()) {
                        printf("        %s* tmp_%s = new %s[(%s) * n];\n",member.GetType().c_str(), name.c_str(), member.GetType().c_str(), member.arraySizes[i].c_str());
                        printf("        is.read(reinterpret_cast<char *>(tmp_%s), sizeof(%s) * ((%s)*n));\n", name.c_str(), member.GetType().c_str(), member.arraySizes[i].c_str());
                        printf("        for (int i = 0; i < %s; ++i) {\n",
                            member.arraySizes[i].c_str());
                    
                        printf("            device->memcpyArrayToGPU<%s>(%s[i], &(tmp_%s[i*n]), n);\n", member.GetType().c_str(), name.c_str(), name.c_str());
                        printf("        }\n");
                        printf("        device->wait();\n");
                        printf("        delete[] tmp_%s;\n", name.c_str());
                } else {
                        printf("        for (int i = 0; i < %s; ++i) {\n",
                            member.arraySizes[i].c_str());
                        printf("            %s[i].Release();\n", name.c_str());
                        printf("            %s[i].deserialize(is, device);\n", name.c_str());
                        printf("        }\n");
                    }
                } else {
                    if(!member.IsSOA()) {
                        printf("        %s* tmp_%s = new %s[n];\n", member.GetType().c_str(), name.c_str(), member.GetType().c_str());
                        printf("        is.read(reinterpret_cast<char *>(tmp_%s), sizeof(%s) * n);\n", name.c_str(), member.GetType().c_str());
                        printf("        device->memcpyArrayToGPU<%s>(%s, tmp_%s, n);\n", member.GetType().c_str(), name.c_str(), name.c_str());
                        printf("        delete[] tmp_%s;\n", name.c_str());
                    } else {
                        printf("        %s.Release();\n", name.c_str());
                        printf("        %s.deserialize(is, device);\n", name.c_str());
                    }
                }
                
                printf("\n");
            }
        }        
        printf("        device->wait();\n");
        printf("    }\n");
        printf("\n");
        // operator[] madness...
        printf("    struct GetSetIndirector {\n");
        if (!soa.templateType.empty()) {
            printf("        OPENPGL_GPU_CALLABLE\n");
            printf("        operator %s<%s>() const {\n", soa.type.c_str(),
                   soa.templateType.c_str());
            printf("            %s<%s> r;\n", soa.type.c_str(), soa.templateType.c_str());
        } else {
            printf("        OPENPGL_GPU_CALLABLE\n");
            printf("        operator %s() const {\n", soa.type.c_str());
            printf("            %s r;\n", soa.type.c_str());
        }
        for (const auto &member : soa.members)
            for (int i = 0; i < member.names.size(); ++i) {
                std::string name = member.names[i];
                if (!member.arraySizes[i].empty()) {
                    printf("            for (int c = 0; c < %s; ++c)\n",
                           member.arraySizes[i].c_str());
                    printf("                r.%s[c] = soa->%s[c][i];\n", name.c_str(),
                           name.c_str());
                } else
                    printf("            r.%s = soa->%s[i];\n", name.c_str(),
                           name.c_str());
            }
        printf("            return r;\n");
        printf("        }\n");

        printf("        OPENPGL_GPU_CALLABLE\n");
        if (!soa.templateType.empty())
            printf("        void operator=(const %s<%s> &a) {\n", soa.type.c_str(),
                   soa.templateType.c_str());
        else
            printf("        void operator=(const %s &a) {\n", soa.type.c_str());
        for (const auto &member : soa.members)
            for (int i = 0; i < member.names.size(); ++i) {
                std::string name = member.names[i];
                if (!member.arraySizes[i].empty()) {
                    printf("            for (int c = 0; c < %s; ++c)\n",
                           member.arraySizes[i].c_str());
                    printf("                soa->%s[c][i] = a.%s[c];\n", name.c_str(),
                           name.c_str());
                } else
                    printf("            soa->%s[i] = a.%s;\n", name.c_str(),
                           name.c_str());
            }
        printf("        }\n\n");
        printf("        SOA *soa;\n");
        printf("        int i;\n");
        printf("    };\n\n");

        printf("    OPENPGL_GPU_CALLABLE\n");
        printf("    GetSetIndirector operator[](int i) {\n");
        //printf("        DCHECK_LT(i, nAlloc);\n");
        printf("        return GetSetIndirector{this, i};\n");
        printf("    }\n");
        printf("    OPENPGL_GPU_CALLABLE\n");
        if (!soa.templateType.empty()) {
            printf("    %s<%s> operator[](int i) const {\n", soa.type.c_str(),
                   soa.templateType.c_str());
            //printf("        DCHECK_LT(i, nAlloc);\n");
            printf("        %s<%s> r;\n", soa.type.c_str(), soa.templateType.c_str());
        } else {
            printf("    %s operator[](int i) const {\n", soa.type.c_str());
            //printf("        DCHECK_LT(i, nAlloc);\n");
            printf("        %s r;\n", soa.type.c_str());
        }
        for (const auto &member : soa.members)
            for (int i = 0; i < member.names.size(); ++i) {
                std::string name = member.names[i];
                if (!member.arraySizes[i].empty()) {
                    printf("        for (int c = 0; c < %s; ++c)\n",
                           member.arraySizes[i].c_str());
                    printf("            r.%s[c] = this->%s[c][i];\n", name.c_str(),
                           name.c_str());
                } else
                    printf("        r.%s = this->%s[i];\n", name.c_str(), name.c_str());
            }
        printf("        return r;\n");
        printf("    }\n");
        printf("\n");


        // Member definitions
        printf("    int nAlloc;\n");
        printf("    bool managed;\n");
        printf("    openpgl::gpu::Device * /*OPENPGL_RESTRICT*/ device; //{nullptr};\n");
        
        for (const auto &member : soa.members) {
            for (int i = 0; i < member.names.size(); ++i) {
                std::string name = member.names[i];
                if (!member.arraySizes[i].empty()) {
                    if (isFlatType(member.type) || member.numPointers > 0)
                        printf("    %s * /*OPENPGL_RESTRICT*/ %s[%s];\n",
                               member.GetType().c_str(), name.c_str(),
                               member.arraySizes[i].c_str());
                    else
                        printf("    SOA<%s> %s[%s];\n", member.type.c_str(), name.c_str(),
                               member.arraySizes[i].c_str());
                } else {
                    if (isFlatType(member.type) || member.numPointers > 0)
                        printf("    %s * /*OPENPGL_RESTRICT*/ %s; //{nullptr};\n", member.GetType().c_str(),
                               name.c_str());
                    else
                        printf("    SOA<%s> %s;\n", member.type.c_str(), name.c_str());
                }
            }
        }

        printf("};\n\n");
    }
}
