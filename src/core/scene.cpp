#include <frozen/unordered_map.h>
#include <pugixml.hpp>

#include "scene.h"

void Scene::parse_from_file(fs::path file_path) {
    // Some code copied from nori:
    // https://github.com/wjakob/nori.git

    pugi::xml_document doc;
    pugi::xml_parse_result ret = doc.load_file(file_path.c_str());

        /* Helper function: map a position offset in bytes to a more readable line/column value */
    auto offset = [&](ptrdiff_t pos) -> std::string {
        std::fstream is(filename);
        char buffer[1024];
        int line = 0, linestart = 0, offset = 0;
        while (is.good()) {
            is.read(buffer, sizeof(buffer));
            for (int i = 0; i < is.gcount(); ++i) {
                if (buffer[i] == '\n') {
                    if (offset + i >= pos)
                        return tfm::format("line %i, col %i", line + 1, pos - linestart);
                    ++line;
                    linestart = offset + i;
                }
            }
            offset += (int) is.gcount();
        }
        return "byte offset " + std::to_string(pos);
    };

    if (!result) /* There was a parser / file IO error */
        throw NoriException("Error while parsing \"%s\": %s (at %s)", filename, result.description(), offset(result.offset));

    enum ETag {
        EScene,
        EFilm,
        ECamera,
        EAccelerator,
        EObject,
        ELight,
        EInvalid
    };


}