#include "SStree.h"

std::vector<Point> SsLeaf::getEntriesCentroids() const {
  return this->points;
}

std::vector<Point> SsInnerNode::getEntriesCentroids() const {
  std::vector<Point> centroids;
  for (const SsNode* child : this->children) {
    centroids.push_back(child->centroid);
  }
  return centroids;
}

void SsLeaf::updateBoundingEnvelope() {
  auto points = getEntriesCentroids();
  centroid = Point(points[0].dim());
  radius = 0;
  for (const Point& point : points) {
    for (size_t i = 0; i < centroid.dim(); ++i) {
      centroid[i] = centroid[i] + point[i];
    }
  }
  centroid = centroid / points.size();
  for (const Point &point : points) {
    if (distance(centroid, point) >= radius) {
      radius = distance(centroid, point);
    }
  }
}

void SsInnerNode::updateBoundingEnvelope() {
  auto points = getEntriesCentroids();

  for (size_t i = 0; i < Settings::k; i++) {
    this->centroid[i] = 0;
    for (auto point : points) {
      this->centroid[i] += point[i];
    }
    this->centroid[i] /= points.size();
  }

  this->radius = 0;
  for (const Point& point : points) {
    this->radius = max(this->radius, distance(this->centroid, point) + this->radius);
  }
}

void SsLeaf::sortEntriesByCoordinate(size_t coordinateIndex){
  sort(points.begin(), points.end(), [coordinateIndex](const Point &p1, const Point &p2) {
    return p1[coordinateIndex] < p2[coordinateIndex]; 
  });
}

void SsInnerNode::sortEntriesByCoordinate(size_t coordinateIndex){
  sort(children.begin(), children.end(), [coordinateIndex](const SsNode *n1, const SsNode *n2) { 
    return n1->centroid[coordinateIndex] < n2->centroid[coordinateIndex]; 
  });
}

NType SsNode::varianceAlongDirection(const std::vector<Point>& centroids, size_t direction) const {
  NType mean = 0;
  NType variance = 0;
  int numPoints = centroids.size();

  // Calculate the mean along the given direction
  for (int i = 0; i < numPoints; ++i) {
    mean += centroids[i][direction];
  }
  mean /= numPoints;

  // Calculate the variance along the given direction
  NType diff;
  for (int i = 0; i < numPoints; ++i) {
    diff = centroids[i][direction] - mean;
    variance += diff * diff;
  }
  variance /= numPoints;

  return variance;
}

size_t SsNode::minVarianceSplit(size_t coordinateIndex) {
  auto m = Settings::m; // Minimum number of points for each partition
  auto M = Settings::M; // Maximum number of points for each partition
  auto points = getEntriesCentroids();

  size_t minSplitIndex = m;
  auto minVarianceSum = NType::max_value();

  // Iterate over possible split indices
  for (size_t splitIndex = m; splitIndex <= M - m; ++splitIndex) {
    std::vector<Point> points1(points.begin(), points.begin() + splitIndex);
    std::vector<Point> points2(points.begin() + splitIndex, points.end());

    auto varianceSum = varianceAlongDirection(points1, coordinateIndex) + varianceAlongDirection(points2, coordinateIndex);

    // Update minimum variance sum and split index if necessary
    if (varianceSum < minVarianceSum) {
      minVarianceSum = varianceSum;
      minSplitIndex = splitIndex;
    }
  }

  return minSplitIndex;
}

void SsNode::sortEntriesByCoordinate(size_t coordinateIndex) {
  auto points = getEntriesCentroids();

  std::sort(points.begin(), points.end(), [coordinateIndex](const Point& p1, const Point& p2) {
    return p1[coordinateIndex] < p2[coordinateIndex];
  });
}

size_t SsNode::directionOfMaxVariance() const {
  size_t maxVarianceDirection = 0;
  NType maxVariance = 0;

  // Iterate over dimensions
  for (size_t direction = 0; direction < dim(); ++direction) {
    NType variance = varianceAlongDirection(getEntriesCentroids(), direction);
    if (variance > maxVariance) {
      maxVariance = variance;
      maxVarianceDirection = direction;
    }
  }

  return maxVarianceDirection;
}

size_t SsNode::findSplitIndex() {
  auto coordinateIndex = directionOfMaxVariance();
  return minVarianceSplit(coordinateIndex);
}

SsNode* SsInnerNode::findClosestChild(const Point& target) const {
  auto minDistance = NType::max_value();
  SsNode* closestChild = nullptr;

  for (SsNode* childNode : children) {
    NType distance = ::distance(childNode->centroid, target);
    if (distance < minDistance) {
      minDistance = distance;
      closestChild = childNode;
    }
  }

  return closestChild;
}

std::pair<SsNode*, SsNode*> SsLeaf::split() {
  size_t splitIndex = findSplitIndex();
  SsNode* newNode1;
  SsNode* newNode2;

  auto points = getEntriesCentroids();

  dynamic_cast<SsLeaf*>(newNode1)->points = std::vector<Point>(this->points.begin(), this->points.begin() + splitIndex);
  dynamic_cast<SsLeaf*>(newNode1)->points = std::vector<Point>(this->points.begin() + splitIndex, this->points.end());

  return std::make_pair(newNode1, newNode2);
}

std::pair<SsNode*, SsNode*> SsInnerNode::split() {
  size_t splitIndex = findSplitIndex();
  SsNode* newNode1;
  SsNode* newNode2;

  auto points = getEntriesCentroids();

  dynamic_cast<SsInnerNode*>(newNode1)->children = std::vector<SsNode*>(this->children.begin(), this->children.begin() + splitIndex);
  dynamic_cast<SsInnerNode*>(newNode1)->children = std::vector<SsNode*>(this->children.begin() + splitIndex, this->children.end());

  return std::make_pair(newNode1, newNode2);
}

SsNode* SsInnerNode::insert(const Point &point) {
  SsNode* closestChild = findClosestChild(point);
  closestChild->updateBoundingEnvelope();
  return closestChild->insert(point);
}

SsNode* SsLeaf::insert(const Point &point) {
  points.push_back(point);
  updateBoundingEnvelope();

  if (points.size() > Settings::M) {
      auto splitNodes = split();
      SsInnerNode* innerNode;
      innerNode->children.push_back(splitNodes.first);
      innerNode->children.push_back(splitNodes.second);
      innerNode->updateBoundingEnvelope();

      splitNodes.first->parent = innerNode;
      splitNodes.second->parent = innerNode;

      if (parent) {
        SsInnerNode* parent = dynamic_cast<SsInnerNode*>(parent);
        for (size_t i = 0; i < parent->children.size(); ++i) {
          if (parent->children[i] == this) {
            parent->children[i] = innerNode;
            break;
          }
        }
        parent->updateBoundingEnvelope();
      }
https://drive.google.com/drive/folders/1oan_whaGrE8QSrLipao8-sRxLNekapbY
      innerNode->parent = parent;

      if (parent){
        parent->updateBoundingEnvelope();
      }

      return innerNode;
  }else {
    if (parent) {
      parent->updateBoundingEnvelope();
    }
    return this;
  }
}

//---------------------------------------------------------------------------------------

SsNode* SsTree::search(SsNode* node, const Point& target) {
  if (node->isLeaf()) {
    if (node->intersectsPoint(target)) {
      return node;
    }
  }else {
    for (SsNode* child : dynamic_cast<SsInnerNode*>(node)->children) {
      if (child->intersectsPoint(target)) {
        auto result = search(child, target);
        if (result != nullptr) {
          return result;
        }
      }
    }
  }
  return nullptr;
}

SsNode* SsTree::searchParentLeaf(SsNode* node, const Point& target) {
  if (node->isLeaf()) {
    return node;
  }else {
    for (SsNode* child : dynamic_cast<SsInnerNode*>(node)->children) {
      if (child->intersectsPoint(target)) {
        return searchParentLeaf(child, target);
      }
    }
  }
  return nullptr;
}

void SsTree::insert(const Point &point) {
  if (root == nullptr) {
      root = new SsLeaf();
      root->insert(point);
      root->updateBoundingEnvelope();
      root->parent = nullptr;
  }else {
    SsNode* insertNode = root->insert(point);
    if (insertNode->parent == nullptr) {
      root = insertNode;
    }
  }
}

void SsTree::insert(Point &point, const std::string &path) {
  point.path = path;
  if (!root) {
    root = new SsLeaf();
    root->insert(point);
    root->updateBoundingEnvelope();
    root->parent = nullptr;
  }else {
    SsNode* insertNode = root->insert(point);
    if (insertNode->parent == nullptr) {
      root = insertNode;
    }
  }
}

std::vector<Point> SsTree::kNNQuery(const Point& center, size_t k) const {
  std::priority_queue<std::pair<NType, const SsNode*>> pq;
  std::vector<Point> result;

  pq.push(std::make_pair(distance(center, root->centroid), root));

  while (!pq.empty() && result.size() < k) {
    const SsNode* currentNode = pq.top().second;
    pq.pop();

    if (currentNode->isLeaf()) {
      const SsLeaf* leaf = dynamic_cast<const SsLeaf*>(currentNode);
      for (const Point& point : leaf->points) {
        pq.push(std::make_pair(distance(center, point), nullptr));
      }
      result.insert(result.end(), leaf->points.begin(), leaf->points.end());

    }else {
      const SsInnerNode* innerNode = dynamic_cast<const SsInnerNode*>(currentNode);
      for (const SsNode* childNode : innerNode->children) {
        pq.push(std::make_pair(distance(center, childNode->centroid), childNode));
      }
    }
  }

  if (result.size() > k) {
    result.resize(k);
  }

  return result;
}

bool SsNode::test(bool isRoot) const {
  size_t count = 0;
  if (this->isLeaf()) {
    const SsLeaf *leaf = dynamic_cast<const SsLeaf *>(this);
    count = leaf->points.size();

    for (const Point &point : leaf->points) {
      if (distance(this->centroid, point) > this->radius) {
        std::cout << "Point outside node radius detected." << std::endl;
        return false;
      }
    }
  }
  else {
    const SsInnerNode *inner = dynamic_cast<const SsInnerNode *>(this);
    count = inner->children.size();

    for (const SsNode *child : inner->children) {
      if (distance(this->centroid, child->centroid) > this->radius) {
        std::cout << "Child centroid outside parent radius detected." << std::endl;
        return false;
      }
      if (!child->test(false)) {
        return false;
      }
    }
  }

  if (!isRoot && (count < Settings::m || count > Settings::M)) {
    std::cout << "Invalid number of children/points detected." << std::endl;
    return false;
  }

  if (!isRoot && !parent) {
    std::cout << "Node without parent detected." << std::endl;
    return false;
  }

  return true;
}
void SsTree::test() const {
  bool result = root->test();

  if (root->parent) {
    std::cout << "Root node parent pointer is not null!" << std::endl;
    result = false;
  }

  if (result) {
    std::cout << "SS-Tree is valid!" << std::endl;
  } else {
    std::cout << "SS-Tree has issues!" << std::endl;
  }
}

void SsNode::print(size_t indent) const {
  for (size_t i = 0; i < indent; ++i) {
    std::cout << "  ";
  }

  // Imprime información del nodo.
  std::cout << "Centroid: " << centroid << ", Radius: " << radius;
  if (isLeaf()) {
    const SsLeaf* leaf = dynamic_cast<const SsLeaf*>(this);
    std::cout << ", Points: [ ";
    for (const Point& p : leaf->points) {
      std::cout << p << " ";
    }
    std::cout << "]";
  } else {
    std::cout << std::endl;
    const SsInnerNode* inner = dynamic_cast<const SsInnerNode*>(this);
    for (const SsNode* child : inner->children) {
      child->print(indent + 1); 
    }
  }
  std::cout << std::endl;
}
void SsTree::print() const {
  if (root) {
    root->print();
  } else {
    std::cout << "Empty tree." << std::endl;
  }
}

void SsLeaf::saveToStream(std::ostream &out) const {
  // Guardar centroid
  auto D = centroid.dim();
  centroid.saveToFile(out, D);

  // Guardar el radio
  float radius_ = radius.getValue();
  out.write(reinterpret_cast<const char*>(&radius_), sizeof(radius_));

  // Guardar el numero de puntos
  size_t numPoints = points.size();
  out.write(reinterpret_cast<const char*>(&numPoints), sizeof(numPoints));

  // Guardar los puntos
  for (const auto& point : points) {
    point.saveToFile(out, D);
  }

  // Guardar las rutas (paths)
  size_t numPaths = paths.size();
  out.write(reinterpret_cast<const char*>(&numPaths), sizeof(numPaths));
  for (const auto& p : paths) {
    size_t pathLength = p.size();
    out.write(reinterpret_cast<const char*>(&pathLength), sizeof(pathLength));
    out.write(p.c_str(), (long) pathLength);
  }
}

void SsInnerNode::saveToStream(std::ostream &out) const {
  // Guardar centroid
  centroid.saveToFile(out, centroid.dim());

  // Guardar el radio
  float radius_ = radius.getValue();
  out.write(reinterpret_cast<const char*>(&radius_), sizeof(radius_));

  // Guardar si apunta a nodos hoja
  bool pointsToLeafs = children[0]->isLeaf();
  out.write(reinterpret_cast<const char*>(&pointsToLeafs), sizeof(pointsToLeafs));

  // Guardar la cantidad de hijos para saber cuántos nodos leer después
  size_t numChildren = children.size();
  out.write(reinterpret_cast<const char*>(&numChildren), sizeof(numChildren));

  // Guardar los hijos
  for (const auto& child : children) {
    child->saveToStream(out);
  }
}

void SsInnerNode::loadFromStream(std::istream &in) {
  // Leer centroid
  auto D = centroid.dim();
  centroid.readFromFile(in, D);

  // leer el valor del radio
  float radius_ = 0;
  in.read(reinterpret_cast<char*>(&radius_), sizeof(radius_));
  this->radius = radius_;

  // leer si apunta a hojas o nodos internos
  bool pointsToLeaf = false;
  in.read(reinterpret_cast<char*>(&pointsToLeaf), sizeof(pointsToLeaf));

  // leer cantidad de hijos
  size_t numChildren;
  in.read(reinterpret_cast<char*>(&numChildren), sizeof(numChildren));

  // leer hijos
  for (size_t i = 0; i < numChildren; ++i) {
    SsNode* child = pointsToLeaf ? static_cast<SsNode*>(new SsLeaf()) : static_cast<SsNode*>(new SsInnerNode());
    child->loadFromStream(in);
    children.push_back(child);
  }
}

void SsLeaf::loadFromStream(std::istream &in) {
  // Leer centroid
  centroid.readFromFile(in, centroid.dim());

  // Leer radio
  float radius_ = 0;
  in.read(reinterpret_cast<char*>(&radius_), sizeof(radius_));
  this->radius = radius_;

  // Leer numero de puntos
  size_t numPoints;
  in.read(reinterpret_cast<char*>(&numPoints), sizeof(numPoints));

  // Leer puntos
  points.resize(numPoints);
  for (size_t i = 0; i < numPoints; ++i) {
    points[i].readFromFile(in, points[i].dim());
  }

  // Leer rutas (paths)
  size_t numPaths;
  in.read(reinterpret_cast<char*>(&numPaths), sizeof(numPaths));
  paths.resize(numPaths);
  for (size_t i = 0; i < numPaths; ++i) {
    size_t pathLength;
    in.read(reinterpret_cast<char*>(&pathLength), sizeof(pathLength));
    char* buffer = new char[pathLength + 1];
    in.read(buffer, (long) pathLength);
    buffer[pathLength] = '\0';
    paths[i] = std::string(buffer);
    delete[] buffer;
  }
}

void SsTree::saveToFile(const std::string &filename) const {
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Cannot open file for writing");
  }

  // Guardar las dimensiones de la estructura
  auto D = root->dim();
  out.write(reinterpret_cast<const char*>(&D), sizeof(D));

  // Guardar si el root es hija o nodo interno
  bool isLeaf = root->isLeaf();
  out.write(reinterpret_cast<const char*>(&isLeaf), sizeof(isLeaf));

  // Guardar el resto de la estructura
  root->saveToStream(out);
  out.close();
}

void SsTree::loadFromFile(const std::string &filename, size_t D) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Cannot open file for reading");
  }
  if (root) {
    delete root;
    root = nullptr;
  }

  // Aquí se asume que el primer valor determina las dimensiones
  in.read(reinterpret_cast<char*>(&D), sizeof(D));

  // El segundo valor determina si el root es hoja
  bool isLeaf;
  in.read(reinterpret_cast<char*>(&isLeaf), sizeof(isLeaf));
  if (isLeaf) {
    root = new SsLeaf();
  } else {
    root = new SsInnerNode();
  }
  root->loadFromStream(in);
  in.close();
}
